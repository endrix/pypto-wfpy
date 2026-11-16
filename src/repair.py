#!/usr/bin/env python3
# coding: utf-8
"""Repair a failing PyPTO kernel named on the command line.

The plain entry point: a kernel path in, a verdict on stdout. The loop itself is
the ``repair_flow`` sub-workflow, shared with repair_softmax.py so it is not
defined twice and cannot drift between the two.

    "Kernel" --> repair_flow() --> "Out"

For the version that reads its input from a file source and ends in a report
viewer, see repair_softmax.py.

Run:
    ./run.sh repair <kernel.py>
    PYPTO_REPAIR_TRANSPORT=acp ./run.sh repair <kernel.py>   # real; spends tokens
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib.repair_config import BUDGET, CONNECTOR, IS_ACP, REPAIR_SKILL, TRANSPORT  # noqa: E402
from src.lib.runner import WF_OUT, default_run_mode  # noqa: E402
from src.workflows.repair_flow import repair_flow  # noqa: E402
from wfpy import connect, run, workflow  # noqa: E402


@workflow(inputs={"Kernel": str}, outputs={"Out": str})
def repair_kernel() -> None:
    """One node: the repair flow, wired straight to the boundary ports.

    The flow's ``Report`` file output is left unconnected on purpose -- the flow
    still writes the report to disk, this entry point just has no viewer to hand
    it to.
    """
    flow = repair_flow()

    connect("Kernel", flow.Kernel)
    connect(flow.Verdict, "Out")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: repair.py <kernel.py>", file=sys.stderr)
        return 2
    print(
        f"[repair] transport={TRANSPORT} budget={BUDGET} mode={default_run_mode()} "
        f"skill={REPAIR_SKILL or '(none)'}",
        file=sys.stderr,
    )
    result = run(
        repair_kernel,
        inputs={"Kernel": sys.argv[1]},
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
