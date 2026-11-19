#!/usr/bin/env python3
# coding: utf-8
"""Sub-workflow: compile a kernel and run it on the Soc CAModel.

    "Kernel" --> Prepare --> RunSim --> "Result"

Two actors, and the whole point of them is the run mode. ``Prepare`` is given
``"sim"`` outright rather than ``default_run_mode()``: on this machine the
default resolves to ``npu``, and a device run compiles the kernel and writes no
cost-model output at all, so every analysis downstream would report SKIP. The
simulator is the instrument here, so the flow pins it.

What that buys, under ``runs/<kernel>_sim/output/<stamp>/CostModelSimulationOutput``:
``_simulate.stat.report.txt`` (pipe usage), ``_simulate.perf.metrics.json``
(tile-op work amounts), ``bubble_analysis.log`` (per-core work and wait) and
``merged_swimlane.json`` (the trace the Toolkit renders).

What it does NOT buy: a measurement. Those cycle counts come from a cost model
running on the CPU. For real numbers the kernel has to run on the NPU under
msprof, which is a different flow.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lib.records import RunResult  # noqa: E402
from src.tasks.prepare import Prepare  # noqa: E402
from src.tasks.run_sim import RunSim  # noqa: E402
from wfpy import connect, workflow  # noqa: E402

RUN_MODE = "sim"


@workflow(inputs={"Kernel": str}, outputs={"Result": RunResult})
def simulate_flow() -> None:
    """Two actors: resolve the kernel into a job, then run it under the model."""
    prepare = Prepare(run_mode=RUN_MODE)
    runner = RunSim()

    connect("Kernel", prepare.Kernel)
    connect(prepare.Job, runner.Job)
    connect(runner.Result, "Result")
