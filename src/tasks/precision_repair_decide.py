"""Route usable checkpoint evidence to the agents, or publish the terminal verdict."""

from pathlib import Path

from wfpy import File, Port, action, guard, task

from src.lib.precision_repair import brief
from src.tasks.precision_repair_session import RepairAssessment


@task
class PrecisionRepairDecide:
    workspace: str
    budget: int

    class Ports:
        Assessment = Port[RepairAssessment](direction="in")
        Evidence = Port[str](direction="out")
        Report = Port[File](direction="out", ext=".md")
        Verdict = Port[str](direction="out")

    def should_retry(self, assessment):
        return assessment.check.retryable and not assessment.check.passed and assessment.attempt < self.budget

    @action(consumes={"Assessment": 1}, produces={"Evidence": 1})
    @guard(lambda self, assessment: self.should_retry(assessment))
    def retry(self, assessment):
        return (f"Repair {assessment.attempt + 1} of {self.budget}.\n\n"
                f"{brief(assessment.check)}\n\nMost recent attempted edit and result:\n{assessment.details}")

    @action(consumes={"Assessment": 1}, produces={"Report": 1, "Verdict": 1})
    @guard(lambda self, assessment: not self.should_retry(assessment))
    def finish(self, assessment):
        ws = Path(self.workspace)
        if assessment.check.passed:
            verdict = "REPAIRED" if assessment.attempt else "ALREADY PASSING"
        else:
            verdict = "FAILED"
        sections = ["# Attention precision repair", "", f"**{verdict}**", "",
                    f"Repair attempts: {assessment.attempt} of {self.budget}.",
                    f"Checked kernel: `{assessment.check.kernel}`", "",
                    (ws / "initial/device-check.md").read_text()]
        for num in range(1, assessment.attempt + 1):
            folder = ws / f"attempt-{num}"
            sections += [f"## Attempt {num}", "", (folder / "review.md").read_text(),
                         (folder / "diagnosis.md").read_text(), (folder / "attempt.md").read_text()]
        report = ws / "precision-repair.md"
        report.write_text("\n\n".join(sections) + "\n")
        return {"Report": str(report), "Verdict": verdict}
