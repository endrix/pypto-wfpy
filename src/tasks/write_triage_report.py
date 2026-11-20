#!/usr/bin/env python3
# coding: utf-8
"""Node: `WriteTriageReport` -- the op table, and the brief the analyst answers."""

from __future__ import annotations

from pathlib import Path

from src.lib.records import Divergence
from src.lib.runner import REPO_ROOT
from wfpy import File, Port, action, task


@task
class WriteTriageReport:
    """Write the comparison, and compose the agent's brief from the same facts.

    Two outputs from one firing because they are one body of evidence read two
    ways: `Report` is the table a person reads, `Brief` is what the analyst is
    asked about. Composing the brief here keeps the agent off the critical path
    of the localisation, which is already finished by this point.
    """

    class Ports:
        Divergence = Port[Divergence](direction="in")
        Report = Port[File](direction="out", ext=".md")
        Brief = Port[str](direction="out")
        Verdict = Port[str](direction="out")

    @action(consumes={"Divergence": 1},
            produces={"Report": 1, "Brief": 1, "Verdict": 1})
    def write(self, div: Divergence) -> dict:
        out_dir = REPO_ROOT / "runs" / "triage"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "triage.md"

        suspect_src = ""
        try:
            suspect_src = Path(div.suspect).read_text(encoding="utf-8")
        except OSError:
            suspect_src = "(source unavailable)"

        table = ["| op | opcode | shape | max abs diff | |", "|---|---|---|---|---|"]
        for d in div.diffs:
            if not d.comparable:
                table.append(f"| {d.magic} | {d.opcode} | {d.shape} | - | {d.note} |")
                continue
            mark = "**first divergence**" if div.culprit and d.magic == div.culprit.magic else ""
            table.append(f"| {d.magic} | {d.opcode} | {d.shape} | {d.max_abs:.6f} | {mark} |")

        if div.error:
            headline = f"TRIAGE FAILED  {div.error}"
        elif div.localized:
            c = div.culprit
            headline = f"CULPRIT  op {c.magic} {c.opcode} {c.shape} (max abs diff {c.max_abs:.6f})"
        else:
            headline = "NO DIVERGENCE  every aligned op agrees; the two builds compute the same thing"

        lines = [
            "# Precision triage", "",
            f"**{headline}**", "",
            f"- reference: `{div.reference}`",
            f"- suspect: `{div.suspect}`",
            f"- ops aligned: {sum(1 for d in div.diffs if d.comparable)} of {len(div.diffs)}",
            "",
            "## Op-by-op comparison", "",
            *table,
            "",
            "## Reading this", "",
            "The ops are in execution order, and the **earliest** difference is the",
            "one that matters. The difference grows as it propagates, so the largest",
            "number is usually the final output and says nothing about the cause.",
            "",
            "Opcodes are tile-level primitives, so one source line becomes several:",
            "a `pypto.softmax` lowers to ROWMAX_SINGLE, SUB, EXP, ROWSUM_SINGLE, DIV.",
            "",
            "Ops before the first divergence are bit-identical between the two",
            "builds, which is what makes the localisation a fact rather than a guess.",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        brief = (
            f"{headline}\n\n"
            f"Reference (trusted): {div.reference}\n"
            f"Suspect (under triage): {div.suspect}\n\n"
            "Op-by-op comparison, in execution order:\n"
            + "\n".join(table)
            + f"\n\n--- suspect source ---\n{suspect_src}"
        )
        return {"Report": str(path), "Brief": brief, "Verdict": headline}
