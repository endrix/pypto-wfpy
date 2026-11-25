"""Exercise bounded repair and the trusted fixture boundary using the wfpy scheduler."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from wfpy import run
from src.lib.precision_repair import PrecisionCheck, SUSPECT, apply_edit, checkpoint_table
from src.lib.records import OpDump
from src.workflows.precision_repair_flow import precision_repair_flow

OLD = "probs = pypto.softmax(scores * SCALE, dim=0)"
NEW = "probs = pypto.softmax(scores * SCALE, dim=-1)"


class PrecisionRepairTest(unittest.TestCase):
    def replay(self, directory, edits, budget=2):
        def evaluate(kernel, folder):
            folder.mkdir(parents=True, exist_ok=True)
            passed = NEW in kernel.read_text()
            report = folder / "device-check.md"
            report.write_text("three seeds passed" if passed else "Aligned NPU checkpoints\nFirst divergent checkpoint: ROWMAX")
            return PrecisionCheck(str(kernel), str(report), passed, not passed)

        def respond(spec, *args, **kwargs):
            if spec.skill == "pypto-op-review":
                outputs = {"Review": "Review maps the softmax call."}
            elif spec.skill == "precision-binary-search":
                outputs = {"Diagnosis": "First mismatch at softmax."}
            else:
                outputs = {"Edit": edits.pop(0)}
            return json.dumps({"outputs": outputs}), [], None, {}

        # Supply minimal local skill files so scheduling also works on CPU-only clones.
        root = directory / "ops"
        for role in ("pypto-op-review", "pypto-precision-debug",
                     "pypto-precision-compare/precision-binary-search"):
            skill = root / role
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(f"---\nname: {skill.name}\ndescription: test role\n---\n")
        with patch.dict("os.environ", WF_SKILLS_ROOT=str(root)), \
                patch("wfpy.runner._invoke_agent", side_effect=respond), \
                patch("src.tasks.precision_repair_session.evaluate", side_effect=evaluate):
            flow = precision_repair_flow(str(directory / "workspace"), budget)
            result = run(flow, inputs={"Suspect": str(SUSPECT)}, out_dir=directory / "wf-out")
        return result["Verdict"], Path(str(result["Report"][0])).read_text()

    def test_invalid_patch_gets_diagnostic_then_repair_passes(self):
        with TemporaryDirectory() as tmp:
            verdict, report = self.replay(Path(tmp), [{"old": "missing", "new": NEW}, {"old": OLD, "new": NEW}])
        self.assertEqual(verdict, ["REPAIRED"])
        for evidence in ("First divergent checkpoint", "Edit rejected", "-    " + OLD,
                         "+    " + NEW, "three seeds passed"):
            self.assertIn(evidence, report)

    def test_budget_exhaustion_cannot_promote_the_last_candidate(self):
        with TemporaryDirectory() as tmp:
            verdict, report = self.replay(Path(tmp), [{"old": OLD, "new": OLD}], budget=1)
        self.assertEqual(verdict, ["FAILED"])
        self.assertIn("Repair attempts: 1 of 1", report)

    def test_same_storage_size_with_different_axes_is_a_structural_fault(self):
        reference = OpDump("10003", "ROWMAX_SINGLE", "[128,1]", "FP32", "/unused")
        suspect = OpDump("10003", "ROWMAX_SINGLE", "[1,128]", "FP32", "/unused")
        _, first = checkpoint_table({"10003": reference}, {"10003": suspect})
        self.assertIn("10003", first)
        self.assertIn("structural difference", first)

    def test_patch_cannot_edit_the_reference_or_fixture(self):
        with TemporaryDirectory() as tmp:
            edit = Path(tmp) / "edit.json"
            edit.write_text(json.dumps({"old": "return 1", "new": "return 0"}))
            with self.assertRaisesRegex(ValueError, "executable line"):
                apply_edit(SUSPECT, edit, Path(tmp) / "candidate.py")
