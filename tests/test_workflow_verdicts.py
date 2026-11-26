import ast
from hashlib import sha256
import json
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.lib import graph, panko, verify
from src.lib.author_config import MIN_ARTIFACT_BYTES, STAGES
from src.lib.frozen_fixture import checked_source
from src.lib.records import JobSpec, OpDump, RunResult, SearchResult
from src.lib.runner import REPO_ROOT, make_job
from src.tasks.dispatch import make_dispatch
from src.tasks.run_selected_kda import RunSelectedKda
from src.tasks.stage_gate import stage_gate
from src.tasks.write_author_report import WriteAuthorReport


class WorkflowVerdictTest(unittest.TestCase):
    def test_stale_artifact_cannot_complete_authoring(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stage, artifact in STAGES:
                submitted = root / f"submitted-{stage}"
                submitted.write_text("# submitted\n" * MIN_ARTIFACT_BYTES)
                gate = stage_gate(stage, artifact, "Stage {verdict}")(workspace=tmp, op="demo")
                brief = gate.check(submitted)
            report = WriteAuthorReport(workspace=tmp, op="demo")
            self.assertTrue(report.write(brief)["Out"].startswith("AUTHORED"))
            for source in ("syntax error!\n" * 100, ""):
                submitted.write_text(source)
                brief = gate.check(submitted)
                self.assertFalse((root / "demo_impl.py").exists())
                # Even an old file copied back after rejection cannot pass this stage.
                (root / "demo_impl.py").write_text("# stale\n" * MIN_ARTIFACT_BYTES)
                self.assertIn("INCOMPLETE", report.write(brief)["Out"])

    def test_checked_candidate_keeps_source_fixture(self):
        source = (REPO_ROOT / "fixtures/02_row_softmax_broken.py").read_text()
        proposed = source.replace("dim=0", "dim=1").replace("atol=1e-3", "atol=1e6")
        checked = ast.parse(checked_source(source, proposed))
        functions = {n.name: n for n in checked.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("dim=1", ast.unparse(functions["row_softmax_kernel"]))
        self.assertIn("atol=0.001", ast.unparse(functions["main"]))
        job = JobSpec("", "", True, "sim", "", ())
        self.assertFalse(job.has_pass_marker("expected PASS: marker"))
        self.assertTrue(job.has_pass_marker("PASS: outputs match"))

    def test_tuning_a_tile_preserves_the_fixture_value_across_rounds(self):
        source = ("VEC_TILE = 16\n@pypto.frontend.jit()\ndef kernel(x): return x * VEC_TILE\n"
                  "def reference(x): return x * VEC_TILE\n")
        checked = checked_source(source, source.replace("VEC_TILE = 16", "VEC_TILE = 32"))
        for expected in (32, 8):
            if expected == 8:
                checked = checked_source(source, checked.replace("_wfpy_tuned_VEC_TILE = 32",
                                                                 "_wfpy_tuned_VEC_TILE = 8"))
            module = ast.parse(checked)
            kernel = next(n for n in module.body if isinstance(n, ast.FunctionDef))
            kernel.decorator_list = []
            env = {}
            exec(compile(module, "<fixture>", "exec"), env)
            self.assertEqual(env["kernel"](1), expected)
            self.assertEqual(env["reference"](1), 16)

    def test_shared_dump_comparator_handles_nonfinite_and_missing_values(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "expected"
            other = Path(tmp) / "actual"
            path.write_bytes(struct.pack("f", 1))
            other.write_bytes(struct.pack("f", float("nan")))
            ref = OpDump("1", "ADD", "[1]", "FP32", str(path))
            def compare(sus):
                return verify.compare({"1": ref}, {"1": sus})
            self.assertIsNotNone(verify.first_divergence(compare(replace(ref, path=str(other)))))
            self.assertTrue(verify.first_divergence(compare(replace(ref, shape="[1,1]"))).structural)
            self.assertFalse(compare(replace(ref, path=str(other) + ".missing"))[0].comparable)

    def test_pool_dispatches_every_kernel_with_fewer_slots(self):
        kernels = ["a", "b", "c", "d", "e"]
        self.assertEqual(make_dispatch(2)().fan(kernels), {"S0": ["a", "c", "e"], "S1": ["b", "d"]})

    def test_each_job_reads_only_its_own_artifacts(self):
        with TemporaryDirectory() as tmp, patch("src.lib.runner.REPO_ROOT", Path(tmp)):
            first = make_job("k.py", "sim")
            second = make_job("k.py", "sim")
            self.assertNotEqual(first.work_dir, second.work_dir)
            (Path(first.work_dir) / "output" / "old").mkdir(parents=True)
            self.assertFalse(RunResult(second, 0, "PASS: fixture", "", 1).sim_output_dir)
            self.assertEqual(Path(second.work_dir).parent.joinpath("latest").resolve(), Path(second.work_dir))

    def test_command_exit_follows_the_deterministic_verdict(self):
        from src import optimize, precision_triage, profile_kernel

        with TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(redirect_stdout(StringIO()))
            stack.enter_context(redirect_stderr(StringIO()))
            stack.enter_context(patch.object(optimize.cann, "apply_to_process"))
            advice = Path(tmp) / "advice.md"
            advice.write_text("Agent explained the failure.")
            for verdict, code in (("BASELINE UNUSABLE", 1), ("NO GAIN", 0)):
                with patch.object(optimize, "run", return_value={"Out": [advice], "Verdict": [verdict]}):
                    self.assertEqual(optimize.main(), code)
            with patch.object(precision_triage, "run", return_value={"Out": [advice], "Verdict": ["INCONCLUSIVE"]}):
                self.assertEqual(precision_triage.main(), 1)
            for result in ([], ["PROFILED  kernel: FAILED its own check"]):
                with patch.object(profile_kernel, "run", return_value={"Out": result}):
                    self.assertEqual(profile_kernel.main(), 1)

    def test_budget_stop_keeps_the_scored_pending_snapshot(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            panko.kernel(root).write_bytes(b"original")
            nodes = root / "optimization/nodes"
            nodes.mkdir(parents=True)
            digest = sha256(b"pending").hexdigest()
            snapshot = nodes / f"{digest}.py"
            snapshot.write_bytes(b"pending")
            tree = {"eval_cache": {digest: {"s": 1, "p": 400}}}
            (nodes.parent / "search_state.json").write_text(json.dumps(tree))
            progress = {"stop_reason": "eval_budget", "refine": {"best": {"code_hash": digest, "p": 400}}}
            self.assertEqual(panko.pending_refinement(root, progress), snapshot)
            snapshot.write_bytes(b"different kernel")
            with self.assertRaisesRegex(ValueError, "scored source"):
                panko.pending_refinement(root, progress)

    def test_empty_ir_viewers_stay_in_the_current_run(self):
        with TemporaryDirectory() as tmp:
            ports = graph.viewer_dumps(tmp)
            self.assertEqual(len(ports), 4)
            for path in ports.values():
                self.assertEqual(Path(path).parent, Path(tmp))
                self.assertIn("No ", json.loads(Path(path).read_text())["note"])

    def test_npu_gate_can_retain_simulator_baseline(self):
        with TemporaryDirectory() as tmp:
            baseline, finalist = (str(Path(tmp) / name) for name in ("base.py", "candidate.py"))
            selected = SearchResult("OPTIMIZED  best=50 cycles", tmp, finalist,
                                    baseline_kernel=baseline, baseline_cycles=100, baseline_dir=tmp)
            job = JobSpec(finalist, "kda", True, "npu", tmp, (), phase="final")
            run = RunResult(job, 0, "PASS: KDA final\nNPU latency: median 3 ms", "", 1)
            with patch("src.tasks.run_selected_kda.evaluate", return_value=run) as checked, \
                    patch("src.tasks.run_selected_kda.compare_kda_pair", return_value=(False, "4/10 wins")):
                result = RunSelectedKda(workspace=tmp, op="kda").check(selected)
            self.assertEqual(checked.call_count, 2)
            self.assertEqual(result.kernel, baseline)
            self.assertTrue(result.verdict.startswith("NO GAIN"))
            self.assertIn("4/10 wins", result.npu_report)
