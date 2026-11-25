"""Keep the planning and verification evidence presented to a branch's developer."""

from hashlib import sha256
import json
from pathlib import Path

from src.lib.residual_contract import (
    ATOL, REFERENCE, RTOL, contract_text, implementation_brief, verification_cases,
)
from wfpy import File, Port, action, task


@task
class ResidualHandoff:
    workspace: str
    filename: str

    class Ports:
        Artifact = Port[File](direction="in")
        Brief = Port[File](direction="out", ext=".md")
        Failed = Port[File](direction="out", ext=".json")

    @action(consumes={"Artifact": 1}, produces={"Brief": 1, "Failed": 1})
    def write(self, artifact):
        workspace = Path(self.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        document = workspace / self.filename
        try:
            content = Path(str(artifact)).read_text(encoding="utf-8")
            if not content.strip():
                raise ValueError(f"empty {self.filename}")
        except (OSError, ValueError) as exc:
            failure = workspace / (document.stem + "_failure.md")
            failure.write_text(f"# Incomplete authoring stage\n\n{self.filename}: {exc}\n")
            result = failure.with_suffix(".json")
            result.write_text(json.dumps({"status": "FAILED", "kernel": "", "report": str(failure)}))
            return {"Failed": str(result)}
        document.write_text(content)
        if self.filename == "PLAN.md":
            (workspace / "SPEC.md").write_text(contract_text() + "\n")
            golden = REFERENCE.read_bytes()
            (workspace / "residual_rmsnorm_golden.py").write_bytes(golden)
            fixture_dir = workspace / "eval"
            fixture_dir.mkdir(exist_ok=True)
            frozen = {"reference": str(REFERENCE), "reference_sha256": sha256(golden).hexdigest(),
                      "atol": ATOL, "rtol": RTOL, "sim": verification_cases("sim"),
                      "npu": verification_cases("npu")}
            (fixture_dir / "test_cases.json").write_text(json.dumps(frozen, indent=2))
        brief = workspace / (document.stem + "_brief.md")
        brief.write_text(implementation_brief(workspace, through=self.filename) + "\n")
        return {"Brief": str(brief)}
