#!/usr/bin/env python3
# coding: utf-8
"""Sub-workflow: the bounded repair loop.

Not a top level -- it is instantiated inside another workflow, which is why it
lives under src/workflows/ rather than at the src/ root.

    "Kernel" --> RunKernel --> Decide --(Done)------> WriteRepairReport --+--> "Report"
                   ^  ^           |--(Exhausted)-----^                    +--> "Verdict"
                   |  |           |
                   |  |        (Repair)
                   |  |           v
                   |  +-- ApplyPatch <-- RepairAgent

A loop in wfpy is an ordinary feedback edge with no backpressure, so it must
terminate by itself: Decide is the only exit, and its three guards are mutually
exclusive so exactly one can fire.

The gate checks more than stdout. A kernel that compiles nothing still exits 0,
so each attempt also dumps IR and the gate requires that a kernel was really
produced and that the IR reached all four levels. Structural change is reported,
not failed: tile shape is a performance choice, not a correctness one.

Two outputs because the caller needs both halves: ``Report`` is a file on disk
for a viewer to open, ``Verdict`` is the one-line text for stdout.

Instantiate it from a parent workflow:

    flow = repair_flow()
    connect(source.Out, flow.Kernel)
    connect(flow.Report, viewer.Report)
    connect(flow.Verdict, "Out")
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.repair_agent import RepairAgent  # noqa: E402
from src.lib.repair_config import BUDGET  # noqa: E402
from src.lib.runner import default_run_mode  # noqa: E402
from src.tasks.apply_patch import ApplyPatch  # noqa: E402
from src.tasks.decide import Decide  # noqa: E402
from src.tasks.run_kernel import RunKernel  # noqa: E402
from src.tasks.write_repair_report import WriteRepairReport  # noqa: E402
from wfpy import File, connect, workflow  # noqa: E402


@workflow(
    inputs={"Kernel": str},
    outputs={
        "Report": File,
        "Verdict": str,
        # The four IR stages of the final kernel, tensor -> execute.
        "TensorGraph": File(ext=".json"),
        "TileGraph": File(ext=".json"),
        "BlockGraph": File(ext=".json"),
        "ExecuteGraph": File(ext=".json"),
    },
)
def repair_flow() -> None:
    """Five actors: run, decide, repair, patch, report."""
    runner = RunKernel(run_mode=default_run_mode())
    decide = Decide(budget=BUDGET)
    repair = RepairAgent()
    patch = ApplyPatch()
    writer = WriteRepairReport()

    connect("Kernel", runner.Initial)
    connect(runner.Result, decide.Result)

    connect(decide.Done, writer.Done)
    connect(decide.Exhausted, writer.Exhausted)

    # The loop: repair -> patch -> back into the runner's second input port,
    # because a wfpy input port does not fan in.
    connect(decide.Repair, repair.Failure)
    connect(repair.Patched, patch.Patched)
    connect(patch.Kernel, runner.Retry)

    connect(writer.Report, "Report")
    connect(writer.Out, "Verdict")

    connect(writer.TensorGraph, "TensorGraph")
    connect(writer.TileGraph, "TileGraph")
    connect(writer.BlockGraph, "BlockGraph")
    connect(writer.ExecuteGraph, "ExecuteGraph")
