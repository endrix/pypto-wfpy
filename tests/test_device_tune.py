"""Exercise the real feedback graph without a paid ACP call or accelerator."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from wfpy import run

from src.lib.device_tune import paired_verdict
from src.tasks.device_tune_round import DeviceTuneRound
from src.workflows.device_tune_flow import make_device_tune_flow


class DeviceTuningTest(unittest.TestCase):
    def test_paired_gate_requires_a_repeatable_gain(self):
        self.assertTrue(paired_verdict([[2] * 10, [1.95] * 10])[0])
        self.assertFalse(paired_verdict([[2] * 10, [1.95] * 8 + [2.1] * 2])[0])
        self.assertFalse(paired_verdict([[2] * 10, [1.999] * 10])[0])
        with self.assertRaises(ValueError):
            paired_verdict([[2] * 10, [float("nan")] * 10])

    def run_loop(self, edits):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        initial = root / "initial.py"
        initial.write_text("TILE = 8\n")
        skills, profiles = [], []
        for name in ("tune-swimlane", "tune-incore"):
            skill = root / name
            skill.mkdir()
            (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test skill\n---\nAnalysis instructions.")
        flow = make_device_tune_flow(str(root), "stub", len(edits), 0)

        def respond(spec, *args, **kwargs):
            reply = edits[len(skills)]
            skills.append(spec.skill)
            return json.dumps({"outputs": {"Artifact": reply}}), [], None, {}

        def capture(controller, kernel, target):
            target.mkdir(parents=True)
            report = target / "profile.md"
            trace = target / "merged_swimlane.json"
            report.write_text("leafHash:913 took 20 us")
            trace.write_text("{}")
            profiles.append((kernel, target))
            return report, trace, ["leafHash:913"]

        def measure(phase, adapter, device, target, *kernels):
            target.mkdir(parents=True)
            (target / "pairs.json").write_text(json.dumps([[10] * 10, [9] * 10]))
            return target

        with patch.dict(os.environ, WF_SKILLS_ROOT=str(root)), patch("wfpy.runner._invoke_agent", side_effect=respond), \
                patch.object(DeviceTuneRound, "_check"), patch.object(DeviceTuneRound, "_profile", capture), \
                patch("src.tasks.device_tune_round.measure", side_effect=measure):
            result = run(flow, inputs={"Kernel": str(initial)}, out_dir=root / "wf-out")
        return skills, profiles, result

    def test_baseline_only_still_checks_and_profiles_the_kernel(self):
        skills, profiles, result = self.run_loop([])
        self.assertEqual(skills, [])
        self.assertEqual(len(profiles), 1)
        self.assertEqual(result["Status"], ["PASS"])

    def test_swimlane_edit_then_targeted_incore_analysis(self):
        edit = {"action": "edit", "old": "TILE = 8", "new": "TILE = 16",
                "evidence": "longest task measured 20 us", "incore_task": "leafHash:913"}
        skills, profiles, result = self.run_loop([edit, {"action": "stop", "evidence": "no further candidate"}])
        self.assertEqual(skills, ["tune-swimlane", "tune-incore"])
        self.assertEqual(len({p for _, p in profiles}), 3)
        self.assertEqual(result["Status"], ["PASS"])
        self.assertIn("TILE = 16", Path(str(result["Winner"][0])).read_text())
        self.assertIn("ADOPTED", Path(str(result["Report"][0])).read_text())

    def test_unsuccessful_edit_exhausts_the_budget(self):
        _, _, result = self.run_loop([{"action": "edit", "evidence": "missing edit"}])
        self.assertEqual(result["Status"], ["FAILED"])
        self.assertIn("REJECTED", Path(str(result["Report"][0])).read_text())
        self.assertIn("TILE = 8", Path(str(result["Winner"][0])).read_text())
