"""Evaluate the initial fault and each proposed one-statement repair."""

from dataclasses import dataclass
from pathlib import Path

from wfpy import File, Port, action, task

from src.lib.precision_repair import PrecisionCheck, apply_edit, evaluate


@dataclass(frozen=True)
class RepairAssessment:
    check: PrecisionCheck
    attempt: int
    details: str


@task
class PrecisionRepairSession:
    workspace: str
    _attempt: int = 0
    _incumbent: PrecisionCheck | None = None

    class Ports:
        Initial = Port[File](direction="in")
        Edit = Port[File](direction="in")
        Review = Port[str](direction="in")
        Diagnosis = Port[str](direction="in")
        Assessment = Port[RepairAssessment](direction="out")
        CallReview = Port[File](direction="out", ext=".md")
        CheckpointDiagnosis = Port[File](direction="out", ext=".md")

    @action(consumes={"Initial": 1}, produces={"Assessment": 1})
    def initial(self, source):
        ws = Path(self.workspace)
        ws.mkdir(parents=True, exist_ok=False)
        candidate = ws / "initial.py"
        candidate.write_bytes(Path(str(source)).read_bytes())
        self._incumbent = evaluate(candidate, ws / "initial")
        return RepairAssessment(self._incumbent, 0, "")

    @action(consumes={"Edit": 1, "Review": 1, "Diagnosis": 1},
            produces={"Assessment": 1, "CallReview": 1, "CheckpointDiagnosis": 1})
    def attempt(self, edit, review, diagnosis):
        self._attempt += 1
        folder = Path(self.workspace) / f"attempt-{self._attempt}"
        folder.mkdir()
        source = Path(self._incumbent.kernel)
        target = folder / "candidate.py"
        (folder / "review.md").write_text(review)
        (folder / "diagnosis.md").write_text(diagnosis)
        artifact = Path(str(edit))
        try:
            diff = apply_edit(source, artifact, target)
        except (ValueError, SyntaxError, OSError) as exc:
            details = f"Edit rejected: {exc}"
        else:
            (folder / "candidate.diff").write_text(diff)
            checked = evaluate(target, folder)
            details = f"```diff\n{diff}\n```\n\n{Path(checked.report).read_text()}"
            if checked.passed or checked.retryable:
                self._incumbent = checked
            else:
                details += "\nThe next repair starts from the last candidate with a usable trace."
        (folder / "attempt.md").write_text(details)
        return {
            "Assessment": RepairAssessment(self._incumbent, self._attempt, details),
            "CallReview": str(folder / "review.md"),
            "CheckpointDiagnosis": str(folder / "diagnosis.md"),
        }
