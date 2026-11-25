"""PANKO protocol, interruption and final-delivery checks."""

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.lib import panko
from src.panko_search import panko_search
from src.tasks.panko_dispatch import PankoDispatch
from wfpy import run


class PankoSearchTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "campaign"
        self.seed = panko.REPO_ROOT / "examples/kda_decay_matrix_verified.py"
        self.skill = self.base / "skills" / panko.MANUAL
        (self.skill / "scripts").mkdir(parents=True)
        (self.skill / "scripts/harness_session.py").touch()
        (self.skill / "SKILL.md").write_text(f"---\nname: {panko.MANUAL}\ndescription: test\n---\n")
        self.enterContext(patch.dict(os.environ, PYPTO_PANKO_COMPARE_KERNEL="", WF_SKILLS_ROOT=str(self.skill.parent),
                                     PYPTO_PANKO_TILE_SEARCH="0", PYPTO_PANKO_BUDGET="2",
                                     PYPTO_PANKO_WORKSPACE=str(self.root)))

    def controller(self):
        return PankoDispatch(workspace=str(self.root), budget=2, device=0, tile_search=False)

    def marker(self, device=0):
        opt = self.root / "optimization"
        opt.mkdir(exist_ok=True)
        digest = sha256(panko.kernel(self.root).read_bytes()).hexdigest()
        tree = {"op": panko.OP, "device_id": device, "best_node": "root",
                "nodes": {"open": [{"id": "u12"}, {"id": "u1"}], "closed": [{"id": "root", "eval": {"s": 1, "p": 125}, "code_hash": digest}]},
                "config": {"eval_budget": 2, "bayesian_optimization_tile_search": False},
                "progress": {"stop_reason": "eval_budget", "_start_ts": 123, "best_latency_us": 125,
                             "preopt_us": 126, "evals_used": 2}}
        (opt / "search_state.json").write_text(json.dumps(tree))
        (opt / "controller-terminal.json").write_text(json.dumps({
            "directive": "STOP", "reason": "eval_budget", "run_start": 123, "delivered_code_hash": digest}))

    def open_tree(self, device=0):
        self.marker(device)
        opt = self.root / "optimization"
        (opt / "controller-terminal.json").unlink()
        path = opt / "search_state.json"
        tree = json.loads(path.read_text())
        tree["progress"]["stop_reason"] = None
        path.write_text(json.dumps(tree))

    def test_resume_coder_then_replay_committed_source(self):
        actor = self.controller()
        self.assertIn("Tune", actor.start(self.seed))
        reply = {"directive": "CODE", "action_id": "u12", "intent": "one edit", "op_file": str(panko.kernel(self.root))}
        self.open_tree()
        self.assertIn("Code", actor.route(json.dumps(reply)))
        with self.assertRaisesRegex(RuntimeError, "active runner"):
            self.controller().start(self.seed)
        actor._lock.close()  # the first process was interrupted
        resumed = self.controller()
        self.assertIn("one edit", resumed.start(self.seed)["Code"])
        self.addCleanup(resumed._lock.close)
        self.assertEqual(json.loads(resumed.apply(str(self.seed))["Tune"])["action_id"], "u12")
        reply["coder"] = False
        self.assertIn("Tune", resumed.route(json.dumps(reply)))

    def test_forged_stop_and_modified_delivery_fail(self):
        actor = self.controller()
        actor.start(self.seed)
        report = actor.route('{"directive":"STOP","reason":"success"}')["Report"]
        self.assertIn("**INCOMPLETE**", Path(report).read_text())
        self.marker()
        bad = self.controller()
        panko.kernel(self.root).write_text("pass\n")
        self.assertIn("do not match", Path(bad.start(self.seed)["Report"]).read_text())

    def test_scheduler_runs_optimizer_and_coder_and_gates_stop(self):
        from src.lib.runner import DEVICES
        roles = []

        def reply(spec, *args, **kwargs):
            roles.append(spec.skill)
            if len(roles) == 1:
                self.open_tree(DEVICES[0])
                outputs = {"Directive": {"directive": "CODE", "op_file": str(panko.kernel(self.root)),
                                         "intent": "preserve seed", "action_id": "u1"}}
            elif len(roles) == 2:
                outputs = {"Source": panko.kernel(self.root).read_text()}
            else:
                self.marker(DEVICES[0])
                outputs = {"Directive": {"directive": "STOP", "reason": "eval_budget"}}
            return json.dumps({"outputs": outputs}), [], None, {}

        (self.base / "stdout.log").write_text("NPU latency: median 1.0 ms\n")
        with patch("wfpy.runner._invoke_agent", side_effect=reply), \
                patch.object(panko, "verify_winner", return_value=str(self.base)) as final:
            result = run(panko_search, out_dir=self.base / "wf-out")
        self.assertIn("**VERIFIED**", Path(str(result["Out"][0])).read_text())
        self.assertEqual(roles, [panko.MANUAL, None, panko.MANUAL])
        self.assertEqual(final.call_count, 1)

    def test_remove_only_the_measurement_flag(self):
        source = b'@pypto.frontend.jit(debug_options={"runtime_debug_mode": 1}, new_ir=False)\ndef f(): pass\n'
        self.assertEqual(panko.without_profiling(source), source.replace(b": 1", b": 0"))
        with self.assertRaisesRegex(ValueError, "single measured"):
            panko.without_profiling(b"pass\n")


class InstalledPankoHarnessTest(unittest.TestCase):
    def test_device_body_is_editable_but_its_host_factory_is_frozen(self):
        import sys

        if not (panko.ops_root() / panko.MANUAL / "scripts/panko_harness.py").is_file():
            self.skipTest("can only check the wrapper contract with the vendor harness installed")
        source = (panko.REPO_ROOT / "examples/kda_decay_matrix_verified.py").read_text()
        with TemporaryDirectory() as temp, patch.dict(sys.modules):
            scripts = panko.harness_root(Path(temp)) / "scripts"
            with patch.object(sys, "path", [str(scripts), *sys.path]):
                import panko_harness
                signature = panko_harness.wrapper_signature
                for before, after, allowed in (
                    ("submit_before_loop=True", "submit_before_loop=False", True),
                    ("rows, features = a.shape", "rows, features = k.shape", False),
                    ("return kda_decay_matrix_codex_kernel\n", "return None\n", False),
                    ("new_ir=False", "new_ir=True", False),
                ):
                    with self.subTest(change=before):
                        self.assertIn(before, source)
                        self.assertEqual(signature(source.replace(before, after)) == signature(source), allowed)
