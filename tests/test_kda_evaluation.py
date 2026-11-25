"""Regression cases for the KDA timeout, score and redesign failures from the IDE run."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.lib import profile
from src.lib.jit_options import add_option
from src.lib.kda_eval import diagnostic
from src.lib.records import JobSpec, RunResult, SearchResult
from src.lib.runner import execute_job
from src.tasks.collect_perf import CollectPerf
from src.tasks.outer_score import OuterScore
from src.tasks.score_perf import ScorePerf
from src.tasks.run_selected_kda import RunSelectedKda
from src.tasks.write_opt_report import WriteOptReport
from src.tasks.verify_kda import VerifyKda


class KdaEvaluationTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.impl = self.root / "kda_impl.py"
        self.impl.write_text("@pypto.frontend.jit()\ndef kernel(): pass\n")
        self.job = JobSpec(str(self.impl), "kda_impl", True, "sim", tmp.name,
                           ("driver", "score"), phase="score")

    def component(self, name, cycles):
        report = self.root / "output" / name / "CostModelSimulationOutput"
        report.mkdir(parents=True)
        (report / "_simulate.stat.report.txt").write_text(f"Total Cycles....: {cycles}\n")
        return str(report)

    def test_collect_and_restart_after_unusable_design(self):
        dirs = (self.component("guard", 1203), self.component("compute", 66000))
        self.component("another-case", 1000000)
        run = RunResult(self.job, 0, "PASS: KDA score", "", 1, report_dirs=dirs)
        self.assertFalse(replace(run, stdout="PASS: internal self-check").passed)
        collect = CollectPerf()
        base = collect.collect(run)
        self.assertEqual(base.total_cycles, 67203)
        self.assertEqual(base.sim_dir, dirs[1])

        score = ScorePerf(budget=1)
        broken = collect.collect(replace(run, rc=-2, error="timeout"))
        score.repair_budget = 1
        self.assertEqual(score._route(broken), "diagnose")
        self.assertIn(str(self.root), score.diagnose(broken))
        self.assertEqual(score._route(broken), "abort")
        self.assertIn("BASELINE UNUSABLE", score.abort(broken).verdict)

        self.assertEqual(score._route(base), "ask")
        self.assertIn("67,203", score.ask_planner(base)["Request"])
        candidate = replace(self.job, kernel=str(self.root / "candidate.py"))
        faster = replace(base, result=replace(run, job=candidate), total_cycles=66091)
        winner = score.stop(faster)
        self.assertEqual(winner.kernel, candidate.kernel)
        verdict = winner.verdict
        self.assertIn("OPTIMIZED  best=66,091", verdict)
        self.assertNotIn("no data", verdict)
        self.assertEqual(score._route(base), "ask")

        Path(dirs[0], "_simulate.stat.report.txt").unlink()
        self.assertFalse(collect.collect(run).found)

    def test_selected_kernel_must_pass_final_npu_before_reporting_a_gain(self):
        winner = str(self.root / "candidates/candidate1.py")
        selection = SearchResult(f"OPTIMIZED  best=450 cycles  kernel={winner}", str(self.root), winner)
        actor = RunSelectedKda(workspace=str(self.root), op="kda")
        writer = WriteOptReport(directory=str(self.root))
        job = replace(self.job, kernel=winner, phase="final", run_mode="npu")
        passed = RunResult(job, 0, "NPU latency: median 0.50 ms\nPASS: KDA final", "", 1)
        failed = replace(passed, rc=1, stdout="", error="fixture mismatch")
        with patch("src.tasks.run_selected_kda.evaluate", side_effect=[passed, failed]) as evaluate:
            for expected in (True, False):
                result = actor.check(selection)
                evaluate.assert_called_with("final", winner, str(self.root), "kda")
                report = writer.write(result)["Report"]
                self.assertEqual(OuterScore._found_gain(report), expected)
                self.assertIn("NPU latency:" if expected else "fixture mismatch", Path(report).read_text())
            empty = SearchResult("BASELINE UNUSABLE", str(self.root))
            self.assertEqual(actor.check(empty), empty)
            self.assertEqual(evaluate.call_count, 2)

    def test_timeout_keeps_fixture_streams_and_partial_reports(self):
        report = self.component("guard", 10)
        script = ("import sys,time; print('START fixture',flush=True); "
                  "print('partial diagnostic',file=sys.stderr,flush=True); time.sleep(15)")
        job = replace(self.job, cmd=(sys.executable, "-c", script))
        with patch("src.lib.runner.cann.environ", return_value={}):
            result = execute_job(job, 1)
        self.assertEqual(result.rc, -2)
        self.assertEqual(result.report_dirs, (report,))
        self.assertIn("Last fixture: START fixture", diagnostic(result))
        self.assertEqual((self.root / "stdout.log").read_text(), result.stdout)
        self.assertIn("partial diagnostic", (self.root / "stderr.log").read_text())

    def test_verify_exhaustion_and_reset(self):
        task = VerifyKda(workspace=str(self.root), op="kda", stage="develop")
        fail = RunResult(self.job, 1, "", "precision mismatch", 1)
        with patch("src.tasks.verify_kda.evaluate", return_value=fail):
            self.assertIn("Retry", task.verify("design"))
            done = Path(task.verify("first repair")["Failed"])
            self.assertIn("VERIFICATION FAILED", done.read_text())
            self.assertIn("Retry", task.verify("next design"))

    def test_redesign_reads_only_matching_passed_profile_of_winner(self):
        actor = OuterScore(budget=1, workspace=str(self.root), op="kda", torch_source="original.py")
        profiles = self.root / "runs/profile"
        profiles.mkdir(parents=True)
        report = self.root / "search.md"
        winner = self.root / "candidate1.py"
        winner.write_text(self.impl.read_text() + "# repaired\n")
        report.write_text(f"**NO GAIN  best=50 cycles  kernel={winner}**\n")
        profile_file = profiles / "candidate1_profile.md"
        def profile_text(source, name, status):
            edited = add_option(source, "debug_options", profile.DEBUG_OPTIONS, note=profile.DEBUG_NOTE)
            digest = sha256(edited.encode()).hexdigest()
            return f"**PROFILED  {name}: {status}**\n- instrumented SHA-256: `{digest}`\nobserved-time"
        with patch("src.tasks.outer_score.REPO_ROOT", self.root):
            for source, name, status in [(self.impl.read_text(), "candidate1", "passed"),
                                         (winner.read_text(), "other", "passed"),
                                         (winner.read_text(), "candidate1", "FAILED its own check")]:
                profile_file.write_text(profile_text(source, name, status))
                self.assertNotIn("observed-time", Path(actor._evidence(report, "test")).read_text())
            profile_file.write_text(profile_text(winner.read_text(), "candidate1", "passed"))
            self.assertIn("observed-time", Path(actor._evidence(report, "test")).read_text())
