#!/usr/bin/env python3
# coding: utf-8
"""Phase 3 -- read the simulator's performance data and name the bottleneck.

The sim run in phase 1 already writes a Soc CAModel report, a per-pipe usage
breakdown and a tile-op work profile, and nothing was reading them. This phase
parses those files and classifies which pipe the kernel is actually limited by.

    "Kernel" --> Prepare --> RunSim --> CollectPerf --> Analyze --> "Report"

Inputs it reads, under <work_dir>/output/<stamp>/CostModelSimulationOutput/:
  _simulate.stat.report.txt    total cycles, per-pipe usage %, task counts
  _simulate.perf.metrics.json  per-tile-op work amounts (longest-path profile)

Run:
    ./run.sh perf
    ./run.sh perf examples/03_fused_linear.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nodes import Prepare, RunResult, RunSim, default_run_mode  # noqa: E402
from wfpy import Port, action, connect, run, task, workflow  # noqa: E402

DEFAULT_KERNELS = [
    "examples/01_fused_elementwise.py",
    "examples/02_row_softmax.py",
    "examples/03_fused_linear.py",
]

# "Average MTE_IN Usage............................:   4387.00   19.86%"
# "Total Cycles....................................:     22087"
STAT_RE = re.compile(r"^\s*\|?-*\s*(.+?)\.{2,}:\s*([\d.]+)(?:\s+([\d.]+)%)?\s*$")

# The pipes a tile kernel can be limited by, in report order.
PIPES = ("Cube", "Vector", "MTE_IN", "MTE1", "MTE_OUT")


@dataclass(frozen=True)
class PerfData:
    result: RunResult
    found: bool
    sim_dir: str
    total_cycles: float = 0.0
    pipe_pct: dict = field(default_factory=dict)   # pipe -> usage %
    pipe_cycles: dict = field(default_factory=dict)  # pipe -> avg cycles
    counters: dict = field(default_factory=dict)   # other scalar counters
    tile_ops: dict = field(default_factory=dict)   # op name -> work amount


def _parse_stat_report(path: Path) -> tuple[float, dict, dict, dict]:
    """Parse the MachineModel text report into (cycles, pipe%, pipe_cycles, counters)."""
    total_cycles = 0.0
    pipe_pct: dict[str, float] = {}
    pipe_cycles: dict[str, float] = {}
    counters: dict[str, float] = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        matched = STAT_RE.match(line)
        if not matched:
            continue
        label, value, pct = matched.group(1).strip(), float(matched.group(2)), matched.group(3)

        # "Total Cycles" appears once per section; the first is the whole run.
        if label == "Total Cycles" and not total_cycles:
            total_cycles = value

        pipe = next((p for p in PIPES if label == f"Average {p} Usage"), None)
        if pipe is not None:
            pipe_cycles[pipe] = value
            if pct is not None:
                pipe_pct[pipe] = float(pct)
        elif pct is None:
            counters.setdefault(label, value)

    return total_cycles, pipe_pct, pipe_cycles, counters


def _parse_tile_ops(path: Path) -> dict:
    """Pull per-tile-op work amounts out of the perf metrics JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    paths = (data.get("summary") or {}).get("tileOpLongestPaths") or {}
    return {
        name: info.get("amountLp", 0)
        for name, info in paths.items()
        if isinstance(info, dict)
    }


@task
class CollectPerf:
    """Locate and parse the simulator's performance artifacts."""

    class Ports:
        Result = Port[RunResult](direction="in")
        Perf = Port[PerfData](direction="out")

    @action(consumes={"Result": 1}, produces={"Perf": 1})
    def collect(self, result: RunResult) -> PerfData:
        base = result.sim_output_dir
        if not result.passed or not base:
            return PerfData(result=result, found=False, sim_dir=base)

        sim_dir = Path(base) / "CostModelSimulationOutput"
        stat = sim_dir / "_simulate.stat.report.txt"
        metrics = sim_dir / "_simulate.perf.metrics.json"
        if not stat.is_file():
            return PerfData(result=result, found=False, sim_dir=str(sim_dir))

        cycles, pipe_pct, pipe_cycles, counters = _parse_stat_report(stat)
        return PerfData(
            result=result,
            found=True,
            sim_dir=str(sim_dir),
            total_cycles=cycles,
            pipe_pct=pipe_pct,
            pipe_cycles=pipe_cycles,
            counters=counters,
            tile_ops=_parse_tile_ops(metrics) if metrics.is_file() else {},
        )


@task
class Analyze:
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

        interesting = (
            "Total Submit Tasks", "Cube Tasks", "Vector Tasks",
            "Average Task Cycles", "Max Task Cycles", "Min Task Cycles",
            "Core Machine Count", "Cube Count", "Vector Count",
        )
        rows = [(k, v) for k, v in perf.counters.items() if k in interesting]
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


@workflow(inputs={"Kernel": str}, outputs={"Report": str})
def analyze_perf() -> None:
    """Wire the four perf phases together."""
    prepare = Prepare(run_mode=default_run_mode())
    runner = RunSim()
    collect = CollectPerf()
    analyze = Analyze()

    connect("Kernel", prepare.Kernel)
    connect(prepare.Job, runner.Job)
    connect(runner.Result, collect.Result)
    connect(collect.Perf, analyze.Perf)
    connect(analyze.Report, "Report")


def main() -> int:
    kernels = sys.argv[1:] or DEFAULT_KERNELS
    failures = 0
    for kernel in kernels:
        result = run(analyze_perf, inputs={"Kernel": kernel})
        for line in result["Report"] or ["(no output)"]:
            print(line)
            if line.startswith(("SKIP", "(no")):
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
