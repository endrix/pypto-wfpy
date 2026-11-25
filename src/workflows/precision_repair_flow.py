"""Trace → review calls → bisect checkpoints → change one statement → rerun."""

import os

from wfpy import File, connect, workflow

from src.agents.diagnose_precision import Diagnose
from src.agents.precision_repair import Repair
from src.agents.review_precision_calls import ReviewCalls
from src.lib.skill_paths import ops_root
from src.viewers.markdown_viewer import MarkdownViewer
from src.tasks.precision_repair_decide import PrecisionRepairDecide
from src.tasks.precision_repair_session import PrecisionRepairSession


def precision_repair_flow(workspace: str, budget: int):
    if budget < 0:
        raise ValueError("PYPTO_PRECISION_REPAIR_BUDGET must be nonnegative")
    root = ops_root()
    roots = [p for p in os.environ.get("WF_SKILLS_ROOT", "").split(os.pathsep) if p]
    for path in (root, root / "pypto-precision-compare"):
        if path.is_dir() and str(path) not in roots:
            roots.append(str(path))
    if roots:
        os.environ["WF_SKILLS_ROOT"] = os.pathsep.join(roots)

    @workflow(inputs={"Suspect": File(ext=".py")},
              outputs={"Report": File(ext=".md"), "Verdict": str})
    def repair_attention():
        session = PrecisionRepairSession(workspace=workspace)
        decide = PrecisionRepairDecide(workspace=workspace, budget=budget)
        review, diagnose, repair = ReviewCalls(), Diagnose(), Repair()
        review_view = MarkdownViewer()
        diagnosis_view = MarkdownViewer()

        connect("Suspect", session.Initial)
        connect(session.Assessment, decide.Assessment)
        connect(decide.Evidence, review.Evidence)
        connect(decide.Evidence, diagnose.Evidence)
        connect(decide.Evidence, repair.Evidence)
        connect(review.Review, diagnose.Review)
        connect(diagnose.Diagnosis, repair.Diagnosis)
        connect(repair.Edit, session.Edit)
        connect(review.Review, session.Review)
        connect(diagnose.Diagnosis, session.Diagnosis)
        connect(session.CallReview, review_view.In)
        connect(session.CheckpointDiagnosis, diagnosis_view.In)
        connect(decide.Report, "Report")
        connect(decide.Verdict, "Verdict")

    return repair_attention
