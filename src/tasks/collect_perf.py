#!/usr/bin/env python3
# coding: utf-8
"""Locate and parse the simulator's performance artifacts."""

from __future__ import annotations

from pathlib import Path

from src.lib.sim_report import parse_stat_report, parse_tile_ops
from src.lib.records import PerfData, RunResult
from wfpy import Port, action, task


@task
class CollectPerf:
    """Read the MachineModel report and tile-op profile for one run."""

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

        cycles, pipe_pct, pipe_cycles, counters = parse_stat_report(stat)
        return PerfData(
            result=result,
            found=True,
            sim_dir=str(sim_dir),
            total_cycles=cycles,
            pipe_pct=pipe_pct,
            pipe_cycles=pipe_cycles,
            counters=counters,
            tile_ops=parse_tile_ops(metrics) if metrics.is_file() else {},
        )
