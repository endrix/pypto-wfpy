#!/usr/bin/env python3
# coding: utf-8
"""Name the limiting pipe and write a perf report."""

from __future__ import annotations

from pathlib import Path

from src.lib.sim_report import INTERESTING
from src.lib.records import PerfData
from wfpy import File, Port, action, task


@task
class AnalyzePerf:
    """Name the limiting pipe and write a perf report."""

    class Ports:
        Perf = Port[PerfData](direction="in")
        # Two outputs for one analysis: the text a terminal run prints, and the
        # file a viewer opens. A viewer needs a path on disk, so a summary that
        # only ever existed as a string would leave it with nothing to show.
        Summary = Port[str](direction="out")
        Report = Port[File](direction="out", ext=".md")

    def _skip(self, job, why: str) -> dict:
        """Report a run with no perf data, still filling both ports.

        The report file is written even here. Both ports are unconditional
        outputs, so a firing that emitted only the summary would stall the
        viewer downstream of it.
        """
        path = Path(job.work_dir) / "perf_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# Perf report -- {job.name}\n\nNo performance data: {why}\n", encoding="utf-8"
        )
        return {"Summary": f"SKIP  {job.name}: {why}", "Report": str(path)}

    @action(consumes={"Perf": 1}, produces={"Summary": 1, "Report": 1})
    def analyze(self, perf: PerfData) -> dict:
        job = perf.result.job
        if not perf.result.passed:
            return self._skip(job, "kernel did not pass, no perf data worth reading")
        if not perf.found:
            return self._skip(job, f"no simulator report under {perf.sim_dir or '(none)'}")

        ranked = sorted(perf.pipe_pct.items(), key=lambda kv: kv[1], reverse=True)
        top, top_pct = ranked[0] if ranked else ("?", 0.0)
        # MTE pipes move data; Cube/Vector do arithmetic. Which one tops the
        # ranking is the difference between memory-bound and compute-bound.
        kind = "memory-bound" if top.startswith("MTE") else "compute-bound"

        lines = [
            f"# Perf report -- {job.name}",
            "",
            f"- kernel: `{job.kernel}`",
            f"- sim dir: `{perf.sim_dir}`",
            f"- total cycles: {perf.total_cycles:,.0f}",
            f"- **limiting pipe: {top} at {top_pct}% -> {kind}**",
            "",
            "## Pipe usage",
            "",
            "| pipe | avg cycles | % of total |",
            "|---|---|---|",
        ]
        for pipe, pct in ranked:
            lines.append(f"| {pipe} | {perf.pipe_cycles.get(pipe, 0):,.0f} | {pct}% |")

        if perf.tile_ops:
            lines += ["", "## Tile-op work profile", "", "| tile op | work amount |", "|---|---|"]
            for name, amount in sorted(perf.tile_ops.items(), key=lambda kv: kv[1], reverse=True):
                lines.append(f"| {name} | {amount:,} |")

        rows = [(k, v) for k, v in perf.counters.items() if k in INTERESTING]
        if rows:
            lines += ["", "## Machine counters", "", "| counter | value |", "|---|---|"]
            lines += [f"| {k} | {v:,.0f} |" for k, v in rows]

        report_path = Path(job.work_dir) / "perf_report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        others = ", ".join(f"{p}={v}%" for p, v in ranked[1:4])
        summary = (
            f"PASS  {job.name:<22} {perf.total_cycles:>8,.0f} cycles  "
            f"limited by {top} ({top_pct}%, {kind})\n"
            f"      next: {others}\n"
            f"      report: {report_path}"
        )
        return {"Summary": summary, "Report": str(report_path)}


