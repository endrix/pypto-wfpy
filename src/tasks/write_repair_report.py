#!/usr/bin/env python3
# coding: utf-8
"""Write the repair run's report to disk, for a viewer to open."""

from __future__ import annotations

from pathlib import Path

from src.lib.records import Outcome
from src.lib.repair_config import BUDGET
from src.lib.runner import REPO_ROOT
from wfpy import File, Port, action, task


@task
class WriteRepairReport:
    """Render the outcome as markdown and emit the file plus a one-line verdict.

    Two outputs: the File feeds the viewer, which can only open something that
    exists on disk, and the string is the workflow's stdout verdict. Done and
    Exhausted are separate input ports because a wfpy input port does not fan in.

    Structural graph deltas are reported rather than failed. Tile shape is a
    performance choice, not a correctness one -- PyPTO splits a reduction across
    tiles correctly -- so a changed leaf count is information for a human, not
    evidence of a bad repair.
    """

    class Ports:
        Done = Port[Outcome](direction="in")
        Exhausted = Port[Outcome](direction="in")
        Report = Port[File](direction="out", ext=".md")
        Out = Port[str](direction="out")

    def _write(self, outcome: Outcome, repaired: bool) -> dict:
        g = outcome.graph
        result = outcome.result
        if not repaired:
            verdict = "GAVE UP"
        elif outcome.attempt == 1:
            # Accepted on the very first run, so the agent was never called:
            # calling that "REPAIRED" would claim credit for a working kernel.
            verdict = "ALREADY OK"
        else:
            verdict = "REPAIRED"

        lines = [
            f"# Repair report -- {result.job.name}",
            "",
            f"- **verdict:** {verdict}",
            f"- kernel: `{result.job.kernel}`",
            f"- runs: {outcome.attempt} (budget: {BUDGET} repair attempt(s))",
            f"- mode: {result.job.run_mode}"
            + (f", device {result.job.device_id}" if result.job.run_mode == "npu" else ""),
            f"- wall time of the last run: {result.seconds}s",
            "",
            "## Why",
            "",
            f"```\n{outcome.why()}\n```",
            "",
            "## What the compiler produced",
            "",
            "| property | value |",
            "|---|---|",
            f"| compiled a kernel | {g.compiled} |",
            f"| IR levels reached | {len(g.levels)}/4 |",
            f"| BLOCK_GRAPH leaves | {g.block_leaves} |",
            f"| distinct tile shapes | {len(g.tile_shapes)} |",
            "",
            f"Levels: {', '.join(sorted(g.levels)) or '(none)'}",
            "",
            f"Tile shapes: {', '.join(sorted(g.tile_shapes)) or '(none)'}",
        ]
        if g.problems():
            lines += ["", "## Compiler problems", ""] + [f"- {p}" for p in g.problems()]
        if g.dump_root:
            lines += ["", f"IR dumps: `{g.dump_root}`"]

        out_dir = REPO_ROOT / "runs" / "repair_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{result.job.name}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        summary = (
            f"{verdict}  {result.job.name}: {outcome.why()}\n"
            f"      graph: compiled={g.compiled} levels={len(g.levels)}/4 "
            f"leaves={g.block_leaves}\n"
            f"      report: {path}"
        )
        return {"Report": str(path), "Out": summary}

    @action(consumes={"Done": 1}, produces={"Report": 1, "Out": 1})
    def on_done(self, outcome: Outcome) -> dict:
        return self._write(outcome, repaired=True)

    @action(consumes={"Exhausted": 1}, produces={"Report": 1, "Out": 1})
    def on_exhausted(self, outcome: Outcome) -> dict:
        return self._write(outcome, repaired=False)
