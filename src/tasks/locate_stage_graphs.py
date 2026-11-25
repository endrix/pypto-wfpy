#!/usr/bin/env python3
# coding: utf-8
"""Node: `LocateStageGraphs` -- one compute-graph dump per IR level, for a viewer."""

from __future__ import annotations

from pathlib import Path
import sys

from src.lib import graph
from src.lib.records import RunResult
from src.lib.runner import REPO_ROOT
from wfpy import File, Port, action, task

# The port that carries each level's dump, tensor -> execute.
LEVELS = (
    ("TENSOR_GRAPH", "TensorGraph"),
    ("TILE_GRAPH", "TileGraph"),
    ("BLOCK_GRAPH", "BlockGraph"),
    ("EXECUTE_GRAPH", "ExecuteGraph"),
)


@task
class LocateStageGraphs:
    """Find the four IR-level dumps a run produced, so each can be rendered.

    The selection logic is `graph.stage_dumps`: the representative `.json` per
    level is chosen by the DUMPING PASS rather than by the level named in a
    dump's own header, because a late pass's dump can still carry an early level
    there and several stages would collapse onto one file. Within a pass the
    largest dump wins -- the whole-kernel graph, not the `___entry___` wrapper.

    Every port is an unconditional output, so each must carry a token on every
    firing. A run that compiled nothing has no dumps of its own, and
    `latest_graph_run` then falls back to the newest run under `runs/` that did
    lower -- a real graph rather than nothing to open. A level that even that
    run never reached falls back to the newest level it did reach, and only a
    tree with no graphs anywhere falls back to the kernel's own source, which at
    least opens.
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
        dumps = graph.stage_dumps(result.job.work_dir)
        source = "this run"
        if result.job.phase and not dumps:
            missing = Path(result.job.work_dir) / "missing_graph.json"
            missing.write_text('{"nodes": [], "edges": [], "note": "No graph from this invocation"}')
            return {port: str(missing) for _, port in LEVELS}
        if not dumps:
            fallback = graph.latest_graph_run(REPO_ROOT / "runs")
            dumps = graph.stage_dumps(fallback) if fallback else {}
            source = f"the newest run that lowered ({fallback})" if dumps else "nothing"

        newest = next(
            (dumps[lv] for lv, _ in reversed(LEVELS) if lv in dumps),
            result.job.kernel,
        )
        print(
            f"[ir] {len(dumps)}/4 IR levels from {source}"
            + ("" if dumps else " -- no graph dumps anywhere under runs/"),
            file=sys.stderr,
        )
        return {port: dumps.get(level, newest) for level, port in LEVELS}
