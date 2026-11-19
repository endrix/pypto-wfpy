#!/usr/bin/env python3
# coding: utf-8
"""Parse the simulator's ``bubble_analysis.log``.

The stat report (see sim_report.py) says which *pipe* is busiest. This file says
something the stat report cannot: how the work was spread over cores, and how
much of each busy core's span was spent waiting rather than computing. A kernel
can be MTE_IN-bound on paper and still be losing most of its time to six of
forty-eight cores sitting idle.

Format, as emitted by the Soc CAModel:

    Top 10 tasks in waiting schedule
        Task:4, label:0-0-4-5-0(), wait:                 5.0238888
    [AIV_25] Execute task num:1
        Core Total Work Time: 6.1405555
        Total Wait Time: 3.3127777
        Wait Schedule Time: 3.3122222
        Wait Predecessor Time: 0.0005555
        Top 3 tasks in waiting schedule time
            Task:1, label:0-0-1-5-0(), wait:             3.3122222

Units are the cost model's own time unit, not cycles and not nanoseconds; only
ratios between them mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

# The core kinds that can actually run a kernel: cube (AIC) and vector (AIV).
COMPUTE_KINDS = ("AIC", "AIV")

# "[AIV_25] Execute task num:1" -- the core's name and how many tasks it ran.
CORE_RE = re.compile(r"^\[(?P<core>[^\]]+)\]\s+Execute task num:(?P<tasks>\d+)\s*$")

# "    Core Total Work Time: 6.1405555" -- one indented metric under a core.
METRIC_RE = re.compile(r"^\s+(?P<key>[A-Z][A-Za-z ]+?):\s*(?P<value>[-\d.]+)\s*$")

# "    Task:4, label:0-0-4-5-0(), wait:   5.0238888"
TASK_RE = re.compile(r"^\s+Task:(?P<task>\d+),\s*label:(?P<label>\S*),\s*wait:\s*(?P<wait>[\d.]+)")

# A core name is "<KIND>_<index>"; the kind is what distinguishes a vector core
# from a cube core or the "Fake Core" placeholder the model emits.
_KIND_RE = re.compile(r"^(?P<kind>.+?)_(?P<index>\d+)$")


@dataclass(frozen=True)
class Core:
    """One core's share of the run."""

    name: str
    tasks: int
    work: float = 0.0
    wait_total: float = 0.0
    wait_schedule: float = 0.0
    wait_predecessor: float = 0.0

    @property
    def kind(self) -> str:
        match = _KIND_RE.match(self.name)
        return match.group("kind") if match else self.name

    @property
    def span(self) -> float:
        """How long this core was involved at all.

        The log's "Core Total Work Time" is the core's whole span, waiting
        INCLUDED -- not its compute time. Confirmed against the vendor's own
        reader, `pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py:38`,
        which derives the real compute time as work minus wait. Adding the wait
        on top of it double-counts and understates every stall.
        """
        return self.work

    @property
    def compute(self) -> float:
        """Time this core actually spent computing."""
        return max(self.work - self.wait_total, 0.0)

    @property
    def utilization_pct(self) -> float:
        return round(100.0 * self.compute / self.span, 1) if self.span else 0.0

    @property
    def bubble_pct(self) -> float:
        """Share of this core's span spent waiting rather than computing."""
        return round(100.0 * self.wait_total / self.span, 1) if self.span else 0.0


@dataclass(frozen=True)
class BubbleData:
    """What ``bubble_analysis.log`` says about one run."""

    found: bool
    path: str = ""
    cores: tuple[Core, ...] = ()
    top_waits: tuple[tuple[int, str, float], ...] = ()  # (task, label, wait)

    @property
    def busy(self) -> tuple[Core, ...]:
        return tuple(c for c in self.compute_cores if c.tasks > 0)

    @property
    def compute_cores(self) -> tuple[Core, ...]:
        """The cores that can run a kernel: the AI cube and vector cores.

        The model also lists a "Fake Core" placeholder and HUB entries. They are
        not compute units, so counting them would dilute every occupancy figure;
        the vendor's reader uses the same AIC+AIV population (72 on this part).
        """
        return tuple(c for c in self.cores if c.kind in COMPUTE_KINDS)

    def by_kind(self) -> dict[str, tuple[int, int]]:
        """kind -> (cores that ran something, cores present)."""
        counts: dict[str, list[int]] = {}
        for core in self.cores:
            row = counts.setdefault(core.kind, [0, 0])
            row[1] += 1
            if core.tasks > 0:
                row[0] += 1
        return {kind: (used, total) for kind, (used, total) in counts.items()}

    @property
    def occupancy_pct(self) -> float:
        """Share of compute cores that ran at least one task."""
        pool = self.compute_cores
        return round(100.0 * len(self.busy) / len(pool), 1) if pool else 0.0

    @property
    def mean_bubble_pct(self) -> float:
        """Average stall share across the cores that actually ran something.

        Only busy cores: an idle core waits for nothing, so including it would
        read as "no stalls" on a badly packed kernel.
        """
        busy = self.busy
        return round(sum(c.bubble_pct for c in busy) / len(busy), 1) if busy else 0.0

    @property
    def mean_utilization_pct(self) -> float:
        """Average utilization over EVERY compute core, idle ones included.

        The fleet-wide number, comparable to the vendor reader's "average core
        utilization": it falls when cores sit idle, which is the point.
        """
        pool = self.compute_cores
        return round(sum(c.utilization_pct for c in pool) / len(pool), 2) if pool else 0.0


# The four indented metrics under a core header, keyed to the Core field they fill.
_FIELDS = {
    "Core Total Work Time": "work",
    "Total Wait Time": "wait_total",
    "Wait Schedule Time": "wait_schedule",
    "Wait Predecessor Time": "wait_predecessor",
}


def parse(log_path: str | Path) -> BubbleData:
    """Read one ``bubble_analysis.log``, or report that it is not there."""
    path = Path(log_path)
    if not path.is_file():
        return BubbleData(found=False, path=str(path))

    cores: list[Core] = []
    top_waits: list[tuple[int, str, float]] = []
    pending: dict[str, float] = {}
    name = ""
    tasks = 0
    # The file opens with a global "Top 10 tasks" block and only then starts
    # listing cores, so task lines before the first core header are the global
    # ranking; task lines after one belong to that core's own "Top 3" block and
    # are already accounted for by its Wait Schedule Time.
    in_core = False

    def flush() -> None:
        if name:
            cores.append(Core(name=name, tasks=tasks, **pending))

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        header = CORE_RE.match(line)
        if header:
            flush()
            name, tasks, pending, in_core = header.group("core"), int(header.group("tasks")), {}, True
            continue
        if in_core:
            metric = METRIC_RE.match(line)
            if metric and metric.group("key") in _FIELDS:
                pending[_FIELDS[metric.group("key")]] = float(metric.group("value"))
                continue
        else:
            task = TASK_RE.match(line)
            if task:
                top_waits.append(
                    (int(task.group("task")), task.group("label"), float(task.group("wait")))
                )
    flush()

    return BubbleData(
        found=bool(cores),
        path=str(path),
        cores=tuple(cores),
        top_waits=tuple(sorted(top_waits, key=lambda row: row[2], reverse=True)),
    )
