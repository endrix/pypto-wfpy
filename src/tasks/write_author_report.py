#!/usr/bin/env python3
# coding: utf-8
"""Node: `WriteAuthorReport` -- what each authoring stage produced."""

from __future__ import annotations

from pathlib import Path

from src.lib.author_config import MIN_ARTIFACT_BYTES, STAGES
from wfpy import File, Port, action, task


@task
class WriteAuthorReport:
    """Say what the five stages left in the workspace.

    One input, the last gate's brief, and the workspace itself as the evidence.
    An earlier version took five scalar `Verdict` ports and joined them; that
    meant five string tokens carrying what the workspace already records, and a
    five-port AND-firing join to put them back together. The workspace is the
    shared state of this flow, so the report reads it.

    It does not raise on a failed stage. The run should end with a document
    naming what is missing, which is more use than a traceback from the stage
    after it.
    """

    workspace: str
    op: str

    class Ports:
        Brief = Port[File](direction="in")
        GoldenFailed = Port[File](direction="in")
        KernelFailed = Port[File](direction="in")
        Report = Port[File](direction="out", ext=".md")
        # The one scalar here, and only because main() prints it to a terminal.
        Out = Port[str](direction="out")

    @action(consumes={"Brief": 1}, produces={"Report": 1, "Out": 1})
    def write(self, brief: str) -> dict:
        return self._report(brief)

    @action(consumes={"GoldenFailed": 1}, produces={"Report": 1, "Out": 1})
    def golden_failed(self, brief: str) -> dict:
        return self._report(brief, "golden")

    @action(consumes={"KernelFailed": 1}, produces={"Report": 1, "Out": 1})
    def kernel_failed(self, brief: str) -> dict:
        return self._report(brief, "develop")

    def _report(self, brief: str, failed_stage: str = "") -> dict:
        ws = Path(self.workspace)
        present, missing = [], []
        for stage, artifact in STAGES:
            name = artifact.format(op=self.op)
            path = ws / name
            if failed_stage == "golden" and stage in ("design", "develop"):
                present.append(f"| {stage} | `{name}` | skipped this round |")
                continue
            size = path.stat().st_size if path.is_file() else 0
            gate = ws / f"brief_after_{stage}.md"
            accepted = gate.is_file() and gate.read_text().startswith(f"Stage OK    {stage}:")
            if not accepted or size < MIN_ARTIFACT_BYTES:
                missing.append(stage)
            present.append(f"| {stage} | `{name}` | {size or '—'} |")

        headline = (
            f"AUTHORED  {self.op}: all five stages produced their artifact"
            if not missing
            else f"INCOMPLETE  {self.op}: no usable artifact from {', '.join(missing)}"
        )

        last_brief = Path(str(brief)).read_text(encoding="utf-8")
        if failed_stage:
            headline = f"INCOMPLETE  {self.op}: {failed_stage} verification failed"
        lines = [
            f"# Authoring {self.op} from PyTorch", "",
            f"**{headline}**", "",
            f"- workspace: `{self.workspace}`",
            f"- last brief: `{brief}`", "",
            "## Artifacts in the workspace", "",
            "| stage | artifact | bytes |", "|---|---|---|",
            *present, "",
            "## What this is", "",
            "Each stage loads one cannbot skill and leaves one declared artifact;",
            "the gate after it checks that artifact exists and is more than a stub,",
            "which is the only thing stopping a stage building on an analysis that",
            "was never written. The graph is the orchestrator -- there is no",
            "orchestrator agent, and no stage decides what happens next.",
            "",
            "A kernel that reaches the last stage is not yet known to work. Run",
            "`./run.sh validate <impl>` to find that out, then `./run.sh profile`",
            "for what the device does with it.",
        ]
        if last_brief.startswith(("# VERIFIED", "# VERIFICATION FAILED")):
            lines = lines[:lines.index("## What this is")] + [last_brief]
        out_dir = ws
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "AUTHORING.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"Report": str(path), "Out": headline}
