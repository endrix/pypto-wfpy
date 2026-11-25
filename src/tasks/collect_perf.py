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
            return PerfData(result=result, found=False, sim_dir="")
        directories = ([Path(p) for p in result.report_dirs] if result.job.phase
                       else [Path(base) / "CostModelSimulationOutput"])
        components = {}
        for directory in directories:
            stat = directory / "_simulate.stat.report.txt"
            if not stat.is_file():
                return PerfData(result=result, found=False, sim_dir=str(directory))
            counters = parse_stat_report(stat)
            if counters[0] <= 0:
                return PerfData(result=result, found=False, sim_dir=str(directory))
            components[str(directory)] = counters

        # Separate JIT calls within one fixture are sequential. Rank by their
        # combined cost; attribute pipe, stall and viewer data to the costliest.
        dominant = max(components, key=lambda path: components[path][0])
        _, pipe_pct, pipe_cycles, counters = components[dominant]
        costs = {path: values[0] for path, values in components.items()}
        return PerfData(
            result=result, found=True, sim_dir=dominant, total_cycles=sum(costs.values()),
            pipe_pct=pipe_pct, pipe_cycles=pipe_cycles, counters=counters,
            tile_ops=parse_tile_ops(Path(dominant) / "_simulate.perf.metrics.json"),
            component_cycles=costs if result.job.phase else {},
        )
