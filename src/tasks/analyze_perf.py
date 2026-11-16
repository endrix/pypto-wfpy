#!/usr/bin/env python3
# coding: utf-8
"""Name the limiting pipe and write a perf report."""

from __future__ import annotations

from pathlib import Path

from src.perf import INTERESTING
from src.records import PerfData
from wfpy import Port, action, task


@task
class AnalyzePerf:
    """Name the limiting pipe and write a perf report."""

    class Ports:
        Perf = Port[PerfData](direction="in")
        Report = Port[str](direction="out")

    @action(consumes={"Perf": 1}, produces={"Report": 1})
    def analyze(self, perf: PerfData) -> str:
        job = perf.result.job
        if not perf.result.passed:
            return f"SKIP  {job.name}: kernel did not pass, no perf data worth reading"
        if not perf.found:
            return f"SKIP  {job.name}: no simulator report under {perf.sim_dir or '(none)'}"

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
        return (
            f"PASS  {job.name:<22} {perf.total_cycles:>8,.0f} cycles  "
            f"limited by {top} ({top_pct}%, {kind})\n"
            f"      next: {others}\n"
            f"      report: {report_path}"
        )


