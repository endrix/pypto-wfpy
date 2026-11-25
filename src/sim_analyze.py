#!/usr/bin/env python3
# coding: utf-8
"""Simulate a PyPTO kernel, analyse the result, show it.

Four nodes, three of which are sub-workflows, so this top level says only what
the run *is* and nothing about how any phase works:

    KernelSource --> simulate_flow --> analyze_flow --+--> visualize_flow
                                                      |
                                                      +--> "Out"

- ``simulate_flow``  compiles the kernel and runs it on the Soc CAModel (2 actors)
- ``analyze_flow``   parses the pipe report, the per-core bubble log, and finds
                     the swimlane trace (4 actors)
- ``visualize_flow`` opens the two reports and renders the swimlane (3 viewers)

Which kernel: ``PYPTO_KERNEL``, else ``argv[1]`` when this file is the program,
else the default below. Not ``argv[1]`` unconditionally -- the source's path is a
constructor parameter, so the graph needs it at BUILD time, and whoever imports
this module owns argv (the wfpy IDE extension passes its own flags, which would
otherwise be read as a kernel path).

Run:
    ./run.sh sim-analyze
    ./run.sh sim-analyze examples/03_fused_linear.py
    PYPTO_KERNEL=examples/04_layer_norm.py ./run.sh sim-analyze
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib import cann  # noqa: E402
from src.lib.runner import REPO_ROOT, WF_OUT  # noqa: E402
from src.sources.kernel_source import KernelSource  # noqa: E402
from src.workflows.analyze_flow import analyze_flow  # noqa: E402
from src.workflows.simulate_flow import RUN_MODE, simulate_flow  # noqa: E402
from src.workflows.visualize_flow import visualize_flow  # noqa: E402
from wfpy import File, connect, run, workflow  # noqa: E402

DEFAULT_KERNEL = "examples/01_fused_elementwise.py"


def _requested_kernel() -> str:
    """Which kernel the source node should hold."""
    from_env = os.environ.get("PYPTO_KERNEL", "").strip()
    if from_env:
        return from_env
    if __name__ == "__main__" and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]
    return DEFAULT_KERNEL


_requested = _requested_kernel()
KERNEL_PATH = _requested if Path(_requested).is_absolute() else str(REPO_ROOT / _requested)


@workflow(inputs={}, outputs={"Out": str})
def sim_analyze() -> None:
    """Four nodes: where the kernel comes from, and the three phases over it."""
    kernel = KernelSource(path=File(KERNEL_PATH))
    simulate = simulate_flow()
    analyze = analyze_flow()
    visualize = visualize_flow()

    connect(kernel.Out, simulate.Kernel)
    connect(simulate.Result, analyze.Result)

    connect(analyze.PerfReport, visualize.PerfReport)
    connect(analyze.BubbleReport, visualize.BubbleReport)
    connect(analyze.Swimlane, visualize.Swimlane)
    for level in ("TensorGraph", "TileGraph", "BlockGraph", "ExecuteGraph"):
        connect(getattr(analyze, level), getattr(visualize, level))

    connect(analyze.Summary, "Out")


def main() -> int:
    # Makes an IDE run behave like a run.sh run: run.sh sources set_env.sh, a
    # direct launch from an IDE does not.
    cann.apply_to_process()
    print(f"[sim] source={KERNEL_PATH}\n[sim] mode={RUN_MODE} (pinned)", file=sys.stderr)

    result = run(sim_analyze, out_dir=str(WF_OUT))
    lines = result["Out"] or ["(no output)"]
    for line in lines:
        print(line)
    return 1 if any(ln.startswith(("SKIP", "(no")) for ln in lines) else 0


if __name__ == "__main__":
    sys.exit(main())
