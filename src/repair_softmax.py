#!/usr/bin/env python3
# coding: utf-8
"""Repair the row-softmax kernel: a file source, the repair flow, a viewer.

Three nodes. The loop itself is a sub-workflow, so this top level says only what
the run *is* -- where the code comes from, what happens to it, and where the
answer goes:

    KernelSource --> repair_flow() --> RepairReportViewer
                           |
                           +--> "Out"

- ``KernelSource`` is a ``@source``: no inputs, so nothing can trigger it and
  nothing can stop it, and wfpy writes the emit-once latch. Its ``path`` is a
  constructor parameter, so one class serves many files. It emits the configured
  ``File`` dataclass rather than a path string -- deliberate, so a source file's
  sibling imports keep resolving -- which downstream nodes coerce with ``str()``.
- ``repair_flow`` is instantiated, not imported as actors: a sub-workflow exposes
  its boundary ports as attributes, so it wires up like any other node.
- ``RepairReportViewer`` opens a file on DISK, which is why the flow emits a
  ``Report`` file alongside the text verdict.

The default input is a deliberately broken softmax: ``pypto.softmax(x, dim=0)``
reduces columns instead of rows. Point it at ``examples/02_row_softmax.py`` and
the loop reports ALREADY OK on the first run without calling the agent.

Run:
    ./run.sh repair-softmax                                 # mock transport
    PYPTO_REPAIR_TRANSPORT=acp ./run.sh repair-softmax      # real; spends tokens
    ./run.sh repair-softmax examples/02_row_softmax.py      # the working kernel
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib.repair_config import BUDGET, CONNECTOR, IS_ACP, REPAIR_SKILL, TRANSPORT  # noqa: E402
from src.lib import cann  # noqa: E402
from src.lib.runner import REPO_ROOT, WF_OUT, default_run_mode  # noqa: E402
from src.sources.kernel_source import KernelSource  # noqa: E402
from src.viewers.pypto_graph_viewer import PyPTOGraphViewer  # noqa: E402
from src.viewers.repair_report_viewer import RepairReportViewer  # noqa: E402
from src.workflows.repair_flow import repair_flow  # noqa: E402
from wfpy import File, connect, run, workflow  # noqa: E402

DEFAULT_KERNEL = "fixtures/02_row_softmax_broken.py"

def _requested_kernel() -> str:
    """Which kernel the source node should hold.

    The source's path is a constructor parameter, so the graph has to know it at
    BUILD time. That cannot come from sys.argv: whoever imports this module owns
    argv, and the wfpy IDE extension passes its own flags -- `sys.argv[1]` would
    then be something like "--workflow" and the graph would be built against a
    path that does not exist. PYPTO_KERNEL is the supported way to choose one;
    argv is honoured only when this file is the program being run.
    """
    from_env = os.environ.get("PYPTO_KERNEL", "").strip()
    if from_env:
        return from_env
    if __name__ == "__main__" and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]
    return DEFAULT_KERNEL


_requested = _requested_kernel()
KERNEL_PATH = _requested if Path(_requested).is_absolute() else str(REPO_ROOT / _requested)


@workflow(inputs={}, outputs={"Out": str})
def repair_softmax() -> None:
    """Three nodes: where the code comes from, what happens to it, where it goes."""
    kernel = KernelSource(path=File(KERNEL_PATH))
    flow = repair_flow()
    view = RepairReportViewer()
    tensor_view = PyPTOGraphViewer()
    tile_view = PyPTOGraphViewer()
    block_view = PyPTOGraphViewer()
    execute_view = PyPTOGraphViewer()

    connect(kernel.Out, flow.Kernel)
    connect(flow.Report, view.Report)
    connect(flow.Verdict, "Out")
    connect(flow.TensorGraph, tensor_view.Graph)
    connect(flow.TileGraph, tile_view.Graph)
    connect(flow.BlockGraph, block_view.Graph)
    connect(flow.ExecuteGraph, execute_view.Graph)


def main() -> int:
    # Makes an IDE run behave like a run.sh run: run.sh sources
    # set_env.sh, a direct launch from an IDE does not.
    cann.apply_to_process()
    print(
        f"[repair] source={KERNEL_PATH}\n"
        f"[repair] transport={TRANSPORT} budget={BUDGET} mode={default_run_mode()} "
        f"skill={REPAIR_SKILL or '(none)'}",
        file=sys.stderr,
    )
    result = run(
        repair_softmax,
        acp_connector=CONNECTOR if IS_ACP else None,
        out_dir=str(WF_OUT),
    )
    lines = result["Out"] or ["(no output)"]
    for line in lines:
        print(line)
    # "ALREADY OK" counts as success: the kernel passed without needing the agent.
    return 0 if any(ln.startswith(("REPAIRED", "ALREADY OK")) for ln in lines) else 1


if __name__ == "__main__":
    sys.exit(main())
