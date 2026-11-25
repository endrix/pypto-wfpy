"""Exercise artifact ownership through wfpy's actual scheduler and file ports."""

from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from wfpy import run
from src.lib.records import JobSpec, RunResult
from src.tasks.outer_score import OuterScore
from src.workflows.author_flow import _PLAN, author_flow


class GoldenFixtureTest(unittest.TestCase):
    def test_blocked_reference_fails_the_inclusive_wide_step(self):
        # A CPU regression for the numerical failure observed on the device.
        with patch.dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"):
            import torch
            from tools.check_kda import check_cases, fixtures
            from torch_src.kda_decay_matrix import decay_matrix

        saved = {}
        with self.assertRaisesRegex(ValueError, r"golden vs FP64: \[1,1\] got nan"):
            check_cases("golden", torch.device("cpu"), "cpu", decay_matrix, None, saved)
        self.assertEqual(list(saved), fixtures("verify")[:-1])


class KdaAuthoringTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name)
        self.source = self.ws / "original.py"
        self.source.write_text("# original operator")
        self.calls = Counter()
        self.phases = []
        self.python = "# generated Python source\n" * 30
        self.malformed = "I’ll run the checks.#!/usr/bin/env python3\n" + self.python

    def replay(self, artifacts, failed_phases=()):
        def respond(spec, *args, **kwargs):
            stage = spec.prompt.split()[0]
            self.calls[stage] += 1
            options = artifacts.get(stage)
            reply = options[self.calls[stage] - 1] if options else self.python
            return json.dumps({"outputs": {"Artifact": reply}}), [], None, {}

        def evaluate(phase, path, workspace, op):
            self.phases.append(phase)
            directory = self.ws / f"evaluation-{len(self.phases)}"
            directory.mkdir()
            job = JobSpec(path, Path(path).stem, True, "npu", str(directory), ("driver", phase), phase=phase)
            if len(self.phases) in failed_phases:
                return RunResult(job, 1, "", f"{phase} mismatch at [1,1]", 1)
            return RunResult(job, 0, f"PASS: KDA {phase}", "", 1)

        # No installed ACP connector, skills or accelerator needed for routing tests.
        plan = [(stage, stage + " " + prompt, None, ext, filename)
                for stage, prompt, _, ext, filename in _PLAN]
        with patch("src.workflows.author_flow._PLAN", plan), patch("wfpy.runner._invoke_agent", side_effect=respond), \
                patch("src.tasks.verify_kda.evaluate", side_effect=evaluate):
            flow = author_flow("kda", str(self.ws), str(self.source), verify_kda=True)
            result = run(flow, inputs={"Source": [str(self.source)]}, out_dir=self.ws / "wf-out")
        return Path(str(result["Report"][-1]))

    def test_malformed_golden_skips_device_and_stops_before_design(self):
        (self.ws / "kda_golden.py").write_text(self.python)  # a previous design
        report = self.replay({"golden": [self.malformed, self.malformed]})
        self.assertEqual(self.calls, {"intent": 1, "api": 1, "golden": 2})
        self.assertEqual(self.phases, [])
        self.assertFalse((self.ws / "kda_golden.py").exists())
        self.assertIn("golden verification failed", report.read_text())
        self.assertIn("SyntaxError at line 1", report.read_text())
        outer = OuterScore(op="kda", workspace=str(self.ws), torch_source=str(self.source), budget=1)
        self.assertFalse(outer._can_redesign_authoring(report))

    def test_numerical_golden_and_syntax_kernel_failures_return_to_their_owners(self):
        report = self.replay({"golden": [self.python, self.python],
                              "develop": [self.malformed, self.python]}, failed_phases={1})
        self.assertEqual(self.calls, {"intent": 1, "api": 1, "golden": 2, "design": 1, "develop": 2})
        self.assertEqual(self.phases, ["golden", "golden", "verify"])
        feedback = [p.read_text() for p in (self.ws / "retries").glob("*.md")]
        self.assertTrue(any("golden mismatch at [1,1]" in text and
                            "Artifact to write: kda_golden.py" in text for text in feedback))
        self.assertTrue(any("SyntaxError at line 1" in text and
                            "Artifact to write: kda_impl.py" in text for text in feedback))
        self.assertIn("**AUTHORED  kda:", report.read_text())

    def test_exhausted_kernel_verification_reports_develop_failure(self):
        report = self.replay({}, failed_phases={2, 3})
        self.assertEqual(self.phases, ["golden", "verify", "verify"])
        self.assertEqual(self.calls["golden"], 1)
        self.assertEqual(self.calls["develop"], 2)
        self.assertIn("develop verification failed", report.read_text())
