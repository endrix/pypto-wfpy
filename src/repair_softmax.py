#!/usr/bin/env python3
# coding: utf-8
"""Repair the row-softmax kernel, fed from a file source and ending in a viewer.

Same repair loop as repair.py, but the kernel under repair arrives from a
``@source`` node holding the Python file, and the verdict is written to disk and
handed to a ``@viewer`` instead of only being printed.

    KernelSource --> RunKernel --> Decide --(Done)------> WriteRepairReport --+--> "Out"
                       ^  ^           |--(Exhausted)-----^                    |
                       |  |           |                                       v
                       |  |        (Repair)                        RepairReportViewer
                       |  |           v
                       |  +-- ApplyPatch <-- RepairAgent

Why the extra two nodes:

- ``KernelSource`` is a ``@source``: no inputs, so nothing can trigger it and
  nothing can stop it, and wfpy writes the emit-once latch. Its ``path`` is a
  constructor parameter, so one class serves many files. At runtime a File token
  is just the resolved path string, which is why it can drive RunKernel's
  ``Port[str]`` input directly.
- ``WriteRepairReport`` exists because a viewer opens a file on DISK. A node that
  only returned a string would leave the viewer with nothing to show, so the
  report is written first and its File token feeds the viewer.

The default input is a deliberately broken softmax: ``pypto.softmax(x, dim=0)``
reduces columns instead of rows. Point it at ``examples/02_row_softmax.py`` and
the loop should accept on the first run without calling the agent at all.

Run:
    ./run.sh repair-softmax                       # mock transport
    PYPTO_REPAIR_TRANSPORT=acp ./run.sh repair-softmax    # real; spends tokens
    ./run.sh repair-softmax examples/02_row_softmax.py    # the working kernel
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.repair_agent import RepairAgent  # noqa: E402
from src.lib.repair_config import BUDGET, CONNECTOR, IS_ACP, REPAIR_SKILL, TRANSPORT  # noqa: E402
from src.lib.runner import REPO_ROOT, WF_OUT, default_run_mode  # noqa: E402
from src.sources.kernel_source import KernelSource  # noqa: E402
from src.tasks.apply_patch import ApplyPatch  # noqa: E402
from src.tasks.decide import Decide  # noqa: E402
from src.tasks.run_kernel import RunKernel  # noqa: E402
from src.tasks.write_repair_report import WriteRepairReport  # noqa: E402
from src.viewers.repair_report_viewer import RepairReportViewer  # noqa: E402
from wfpy import File, connect, run, workflow  # noqa: E402

DEFAULT_KERNEL = "fixtures/02_row_softmax_broken.py"

# Resolved at import time: the source's path is a constructor parameter, so the
# graph has to know it when it is built rather than when it is run.
_requested = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_KERNEL
KERNEL_PATH = _requested if Path(_requested).is_absolute() else str(REPO_ROOT / _requested)


@workflow(inputs={}, outputs={"Out": str})
def repair_softmax() -> None:
    """Seven actors: the repair loop, plus a file source and a report viewer."""
    kernel = KernelSource(path=File(KERNEL_PATH))
    runner = RunKernel(run_mode=default_run_mode())
    decide = Decide(budget=BUDGET)
    repair = RepairAgent()
    patch = ApplyPatch()
    writer = WriteRepairReport()
    view = RepairReportViewer()

    # The source replaces the workflow's boundary input entirely.
    connect(kernel.Out, runner.Initial)
    connect(runner.Result, decide.Result)

    connect(decide.Done, writer.Done)
    connect(decide.Exhausted, writer.Exhausted)

    # The loop: repair -> patch -> back into the runner's second input port.
    connect(decide.Repair, repair.Failure)
    connect(repair.Patched, patch.Patched)
    connect(patch.Kernel, runner.Retry)

    # One output broadcasts: the file goes to the viewer, the text to stdout.
    connect(writer.Report, view.Report)
    connect(writer.Out, "Out")


def main() -> int:
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
