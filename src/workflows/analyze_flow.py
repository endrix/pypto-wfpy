#!/usr/bin/env python3
# coding: utf-8
"""Sub-workflow: read everything the simulator left behind.

    "Result" --> CollectPerf --+--> AnalyzePerf    --> "Summary", "PerfReport"
                               |
                               +--> AnalyzeBubbles --> "BubbleReport"
                               |
                               +--> LocateSwimlane --> "Swimlane"

One parse, three readings of it. ``CollectPerf.Perf`` is a single output port
feeding three consumers, which is fine -- a wfpy output port broadcasts, so each
of the three gets its own copy of the token. (Input ports are the ones that do
NOT fan in; three producers into one port would be the mistake.)

They are separate nodes because they answer different questions and fail
independently:

- ``AnalyzePerf`` ranks the pipes: which hardware unit the cycles went to.
- ``AnalyzeBubbles`` ranks the cores: whether the work was spread at all, and
  how much of each busy core's span was spent waiting. A kernel can be
  MTE_IN-bound on four cores out of forty-eight, and only this says so.
- ``LocateSwimlane`` summarises nothing; it just finds the trace, because what
  reads it is the Toolkit's renderer rather than this workflow.
- ``LocateStageGraphs`` does the same for the compiler's per-pass dumps, one per
  IR level, so the lowering can be looked at and not only the timings.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lib.records import PerfData, RunResult  # noqa: E402
from src.tasks.analyze_bubbles import AnalyzeBubbles  # noqa: E402
from src.tasks.analyze_perf import AnalyzePerf  # noqa: E402
from src.tasks.collect_perf import CollectPerf  # noqa: E402
from src.tasks.locate_stage_graphs import LocateStageGraphs  # noqa: E402
from src.tasks.locate_swimlane import LocateSwimlane  # noqa: E402
from wfpy import File, connect, workflow  # noqa: E402


@workflow(
    inputs={"Result": RunResult},
    outputs={
        "Summary": str,
        # ext= on every boundary output, not just on the producing port: at a
        # sub-workflow boundary the declared ext is the only thing that names the
        # materialized copy, so leaving it off here yields an extensionless file
        # that the IDE opens as plain text.
        "PerfReport": File(ext=".md"),
        "BubbleReport": File(ext=".md"),
        "Swimlane": File(ext=".json"),
        # The parsed counters themselves, for a caller that has to DECIDE on
        # them rather than read them -- the optimization loop scores on
        # total_cycles. Left unconnected by sim_analyze, which only reports.
        "Perf": PerfData,
        # One compute-graph dump per IR level, so the compiler's own view of the
        # kernel can be rendered beside the numbers.
        "TensorGraph": File(ext=".json"),
        "TileGraph": File(ext=".json"),
        "BlockGraph": File(ext=".json"),
        "ExecuteGraph": File(ext=".json"),
    },
)
def analyze_flow() -> None:
    """Four actors: one collector and three independent readings of its output."""
    collect = CollectPerf()
    pipes = AnalyzePerf()
    cores = AnalyzeBubbles()
    swim = LocateSwimlane()
    graphs = LocateStageGraphs()

    connect("Result", collect.Result)
    # The IR dumps come from the run itself, not from the parsed counters.
    connect("Result", graphs.Result)

    connect(collect.Perf, pipes.Perf)
    connect(collect.Perf, cores.Perf)
    connect(collect.Perf, swim.Perf)

    connect(collect.Perf, "Perf")

    connect(pipes.Summary, "Summary")
    connect(pipes.Report, "PerfReport")
    connect(cores.Report, "BubbleReport")
    connect(swim.Swimlane, "Swimlane")

    for level in ("TensorGraph", "TileGraph", "BlockGraph", "ExecuteGraph"):
        connect(getattr(graphs, level), level)
