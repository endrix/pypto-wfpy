#!/usr/bin/env python3
# coding: utf-8
"""Node: `LocateStageGraphs` -- one compute-graph dump per IR level, for a viewer."""

from __future__ import annotations

from src.lib import graph
from src.lib.records import RunResult
from wfpy import File, Port, action, task


@task
class LocateStageGraphs:
    """Find the four IR-level dumps a run produced, so each can be rendered.

    The selection logic is `graph.stage_dumps`: the representative `.json` per
    level is chosen by the DUMPING PASS rather than by the level named in a
    dump's own header, because a late pass's dump can still carry an early level
    there and several stages would collapse onto one file. Within a pass the
    largest dump wins -- the whole-kernel graph, not the `___entry___` wrapper.

    """

    class Ports:
        Result = Port[RunResult](direction="in")
        TensorGraph = Port[File](direction="out", ext=".json")
        TileGraph = Port[File](direction="out", ext=".json")
        BlockGraph = Port[File](direction="out", ext=".json")
        ExecuteGraph = Port[File](direction="out", ext=".json")

    @action(
        consumes={"Result": 1},
        produces={"TensorGraph": 1, "TileGraph": 1, "BlockGraph": 1, "ExecuteGraph": 1},
    )
    def locate(self, result: RunResult) -> dict:
        return graph.viewer_dumps(result.job.work_dir)
