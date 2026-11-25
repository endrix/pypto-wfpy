"""Check one KDA author's artifact and send one repair attempt to its owner."""

from pathlib import Path
from tempfile import mkdtemp

from src.lib.kda_eval import diagnostic, evaluate
from wfpy import File, Port, action, task


@task
class VerifyKda:
    workspace: str
    op: str
    stage: str
    _retried: bool = False

    class Ports:
        Brief = Port[File](direction="in")
        Retry = Port[File](direction="out", ext=".md")
        Passed = Port[File](direction="out", ext=".md")
        Failed = Port[File](direction="out", ext=".md")

    @action(consumes={"Brief": 1}, produces={"Retry": 1, "Passed": 1, "Failed": 1})
    def verify(self, brief: str) -> dict:
        ws = Path(self.workspace)
        phase, suffix = ("golden", "golden") if self.stage == "golden" else ("verify", "impl")
        target = ws / f"{self.op}_{suffix}.py"
        if target.is_file():
            result = evaluate(phase, str(target), self.workspace, self.op)
            directory = Path(result.job.work_dir)
            passed, detail = result.passed, diagnostic(result)
        else:
            # The preceding gate removed a missing, stub or invalid Python artifact.
            root = ws / "evaluations"
            root.mkdir(exist_ok=True)
            directory = Path(mkdtemp(dir=root, prefix=f"{target.stem}-rejected-"))
            passed, detail = False, Path(str(brief)).read_text(encoding="utf-8")

        status = "VERIFIED" if passed else "VERIFICATION FAILED"
        report = directory / "verification.md"
        report.write_text(f"# {status} — {self.stage}\n\nArtifact: {target}\n\n"
                          f"{detail}\n\nAuthoring brief: {brief}\n", encoding="utf-8")
        if not passed and not self._retried:
            self._retried = True
            return {"Retry": str(report)}
        self._retried = False
        return {"Passed" if passed else "Failed": str(report)}
