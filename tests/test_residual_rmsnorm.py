"""Check independent design routing, two-output correctness and held-out selection."""

from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from wfpy import run

from src.lib.residual_contract import BRANCHES
from src import residual_rmsnorm as launcher


class ResidualTest(unittest.TestCase):
    def test_oracle_checks_both_outputs_and_cancellation(self):
        with patch.dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"):
            import torch
            from src.lib.residual_eval import check_case, unfused

        device = torch.device("cpu")
        def correct(x, res, weight, eps, run_mode):
            return unfused(x, res, weight, eps=eps)
        def broken(x, res, weight, eps, run_mode):
            y, z = correct(x, res, weight, eps, run_mode)
            return y, z + 1
        check_case(correct, (5, 32), 1, "sim", device)
        check_case(correct, (5, 32), 1, "sim", device, cancel=True)
        with self.assertRaisesRegex(AssertionError, "residual_sum"):
            check_case(broken, (5, 32), 1, "sim", device, cancel=True)

    def test_holdout_and_noisy_samples_can_keep_unfused(self):
        from tools.compare_residual import select
        samples = {
            "unfused": {"train": [2.0] * 12, "held": [2.0] * 12},
            "row_reuse": {"train": [1.8] * 12, "held": [1.8] * 8 + [2.01] * 4},
            "split_reduction": {"train": [1.1] * 12, "held": [2.5] * 12},
        }
        self.assertEqual(select(samples), "unfused")
        samples["row_reuse"]["held"] = [1.8] * 12
        self.assertEqual(select(samples), "row_reuse")

    def test_scheduler_keeps_designs_separate_and_reports_exhausted_branch(self):
        with TemporaryDirectory() as root, ExitStack() as stack:
            workspace = Path(root)
            calls = Counter()
            def respond(spec, *args, **kwargs):
                if "Implement the design" in spec.prompt:
                    name = next(b for b in BRANCHES if b in spec.prompt)
                    calls[name] += 1
                    text = ("@pypto.frontend.jit()\ndef residual_rmsnorm(): pass\n"
                            if name == "row_reuse" else "broken syntax!")
                else:
                    text = "Use independent row and reduction checks on both outputs."
                return json.dumps({"outputs": {"Artifact": text}}), [], None, {}

            def profile(_adapter, _device, directory, _kernel):
                directory.mkdir(parents=True)
                report = directory / "profile.md"
                report.write_text("CPU test profile")
                trace = directory / "trace.json"
                trace.write_text("{}")
                return report, trace, []

            def compare(paths, ws, device):
                self.assertEqual(set(paths), {"row_reuse"})
                self.assertTrue(Path(paths["row_reuse"]).is_file())
                return {"winner": "unfused", "samples_ms": {}, "errors": {}}, ws, ""

            skills = workspace / "skills"
            for name in ("pypto-op-plan", "pypto-op-design", "pypto-op-verify", "pypto-op-develop"):
                folder = skills / name
                folder.mkdir(parents=True)
                (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test stage\n---\n")
            stack.enter_context(patch.dict(os.environ, PYPTO_DEVICE_TUNE_BUDGET="0",
                                          WF_SKILLS_ROOT=str(skills)))
            stack.enter_context(patch.object(launcher, "REPO_ROOT", workspace))
            stack.enter_context(patch("wfpy.runner._invoke_agent", side_effect=respond))
            stack.enter_context(patch("src.tasks.residual_verify.measure"))
            stack.enter_context(patch("src.tasks.device_tune_round.measure"))
            stack.enter_context(patch("src.tasks.device_tune_round.device_profile", side_effect=profile))
            stack.enter_context(patch("src.tasks.compare_residual.run_comparison", side_effect=compare))
            result = run(launcher.residual_rmsnorm, out_dir=workspace / "wf-out")
            report = Path(str(result["Out"][-1])).read_text()
            self.assertIn("**INCOMPLETE**", report)
            self.assertIn("split_reduction: FAILED", report)
            self.assertIn("row_reuse: PASS", report)
            self.assertEqual(calls, {"row_reuse": 1, "split_reduction": 2})
            [trial] = (workspace / "custom/residual_rmsnorm").iterdir()
            for name in BRANCHES:
                self.assertTrue((trial / name / "VERIFICATION.md").is_file())
                fixtures = json.loads((trial / name / "eval/test_cases.json").read_text())
                self.assertEqual((len(fixtures["sim"]), len(fixtures["npu"])), (2, 5))


if __name__ == "__main__":
    unittest.main()
