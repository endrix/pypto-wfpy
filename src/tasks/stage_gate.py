#!/usr/bin/env python3
# coding: utf-8
"""Node: `stage_gate` -- check a stage's declared artifact, and place it."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from src.lib.author_config import MAX_INLINE, MIN_ARTIFACT_BYTES
from wfpy import File, Port, action, task


def stage_gate(stage: str, artifact: str, next_brief: str) -> type:
    """Build the gate that follows one authoring stage.

    This is the orchestrator's only real job -- "⛔ 核查规则：子技能声明的所有分析
    制品均已生成后，才允许进入 ITER 循环" -- and the only thing stopping a stage
    from building on an analysis that was never written.

    It does two things. It REJECTS an artifact that is missing or too short to be
    one: an agent that replies "I have written SPEC.md" produces a short file,
    and a gate that only checked existence would pass it. And it PLACES the
    artifact in the workspace under its declared name, because the cannbot skills
    read `custom/<op>/SPEC.md` from disk rather than being handed its text -- so
    the workspace, not the token, is what the next stage actually consumes.

    A rejection does not raise. It emits a brief that tells the next stage the
    artifact is missing, so the run ends with a report naming the stage that
    failed rather than a traceback.
    """

    @task
    class StageGate:
        workspace: str
        op: str

        class Ports:
            Artifact = Port[File](direction="in")
            # The brief is a document, so it is a file. The verdict used to be a
            # second, scalar port feeding a five-input join; it is a heading in
            # this document now, and the report reads the workspace for the rest.
            Brief = Port[File](direction="out", ext=".md")

        @action(consumes={"Artifact": 1}, produces={"Brief": 1})
        def check(self, written: str) -> str:
            name = artifact.format(op=self.op)
            target = Path(self.workspace) / name
            source = Path(str(written))

            size = source.stat().st_size if source.is_file() else 0
            syntax_error = ""
            if size and target.suffix == ".py":
                try:
                    compile(source.read_bytes(), str(source), "exec", dont_inherit=True)
                except SyntaxError as exc:
                    syntax_error = f"line {exc.lineno}: {exc.msg}\n{(exc.text or '').rstrip()}"

            if syntax_error:
                verdict = f"INVALID  {stage}: {name} has a SyntaxError at {syntax_error}"
            elif size >= MIN_ARTIFACT_BYTES:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                verdict = f"OK    {stage}: wrote {name} ({size} bytes)"
            elif size:
                verdict = (
                    f"THIN  {stage}: {name} is only {size} bytes, under the "
                    f"{MIN_ARTIFACT_BYTES} an artifact needs -- treating it as absent"
                )
            else:
                verdict = f"MISSING  {stage}: produced no {name}"

            if syntax_error or size < MIN_ARTIFACT_BYTES:
                # A later design must never inherit an earlier design’s accepted file.
                target.unlink(missing_ok=True)

            print(f"[author] {verdict}", file=sys.stderr)

            # The artifact's text goes INTO the brief as well as onto disk. Belt
            # and braces: the stages have file tools, but a chain that only works
            # when a tool call succeeds has a silent failure mode -- a stage would
            # answer from its prompt alone and look like it had read the workspace.
            body = ""
            if size:
                text = source.read_text(encoding="utf-8", errors="replace")
                body = (
                    f"\n\n## {name}, as written\n\n"
                    + (text[:MAX_INLINE] + "\n\n_[truncated]_" if len(text) > MAX_INLINE else text)
                )

            brief_path = Path(self.workspace) / f"brief_after_{stage}.md"
            brief_path.parent.mkdir(parents=True, exist_ok=True)
            brief_path.write_text(
                next_brief.format(
                    workspace=self.workspace, op=self.op, artifact=name, verdict=verdict
                ) + body + "\n",
                encoding="utf-8",
            )
            return str(brief_path)

    StageGate.__name__ = f"{stage.capitalize()}Gate"
    StageGate.__qualname__ = StageGate.__name__
    return StageGate
