#!/usr/bin/env python3
# coding: utf-8
"""Node: `AnalyzeBubbles` -- turn the simulator's bubble log into a stall report."""

from __future__ import annotations

from pathlib import Path

from src.lib import bubbles
from src.lib.records import PerfData
from wfpy import File, Port, action, task


@task
class AnalyzeBubbles:
    """Say how the work was spread over cores and how much of it was waiting.

    This is the half of the analysis the pipe percentages cannot give you. The
    stat report ranks pipes within the cores that ran; this ranks the cores, so
    a kernel that keeps one pipe busy on four cores out of forty-eight is
    visibly under-parallelised rather than merely "memory-bound".
    """

    top_n: int = 8  # has a default -> actor state, not a parameter

    class Ports:
        Perf = Port[PerfData](direction="in")
        Report = Port[File](direction="out", ext=".md")

    @action(consumes={"Perf": 1}, produces={"Report": 1})
    def analyze(self, perf: PerfData) -> str:
        job = perf.result.job
        path = Path(job.work_dir) / "bubble_report.md"
        data = bubbles.parse(Path(perf.sim_dir or ".") / "bubble_analysis.log")

        lines = [f"# Core occupancy and stalls -- {job.name}", ""]
        if not data.found:
            # Written anyway, not skipped: the port is an unconditional output,
            # so a viewer downstream must still be handed a real file. A missing
            # log is itself worth reading about -- it means sim mode did not run.
            lines += [
                f"No bubble analysis at `{data.path}`.",
                "",
                "The simulator writes this file only in sim mode (`RunMode.SIM`); a",
                "device run compiles the kernel and produces no cost-model output.",
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return str(path)

        kinds = data.by_kind()
        lines += [
            f"- log: `{data.path}`",
            f"- compute cores that ran anything: **{len(data.busy)} of "
            f"{len(data.compute_cores)}** ({data.occupancy_pct}%)",
            f"- fleet utilization, idle cores included: **{data.mean_utilization_pct}%**",
            f"- mean stall share on the busy cores: **{data.mean_bubble_pct}%**",
            "",
            "## Occupancy by core kind",
            "",
            "| kind | used | present |",
            "|---|---|---|",
        ]
        for kind, (used, total) in sorted(kinds.items()):
            lines.append(f"| {kind} | {used} | {total} |")

        lines += [
            "",
            "## Busiest cores",
            "",
            "| core | tasks | span | compute | util % | stall % | sched wait | pred wait |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for core in sorted(data.busy, key=lambda c: c.span, reverse=True)[: self.top_n]:
            lines.append(
                f"| {core.name} | {core.tasks} | {core.span:.3f} | {core.compute:.3f} "
                f"| {core.utilization_pct}% | {core.bubble_pct}% "
                f"| {core.wait_schedule:.3f} | {core.wait_predecessor:.3f} |"
            )

        if data.top_waits:
            lines += ["", "## Tasks waiting longest to be scheduled", "",
                      "| task | label | wait |", "|---|---|---|"]
            for task_id, label, wait in data.top_waits[: self.top_n]:
                lines.append(f"| {task_id} | `{label}` | {wait:.3f} |")

        lines += [
            "",
            "## Reading this",
            "",
            "Times are the cost model's own unit -- only ratios between them mean",
            "anything. `span` is how long the core was involved at all and",
            "`compute` is span minus wait, so `util %` and `stall %` sum to 100.",
            "",
            "`sched wait` is time a core spent waiting for work to be dispatched",
            "to it, `pred wait` time spent waiting on a dependency.",
            "Schedule-dominated waiting points at tiling and launch shape;",
            "predecessor-dominated waiting points at the dependency chain.",
            "",
            "Occupancy counts the AIC and AIV cores only -- the model also lists a",
            "\"Fake Core\" placeholder and HUB entries, which are not compute units.",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)
