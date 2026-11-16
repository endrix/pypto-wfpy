#!/usr/bin/env python3
# coding: utf-8
"""A bounded repair loop around a failing PyPTO kernel.

    "Kernel" --> RunKernel --> Decide --(Done)-------> RepairReport --> "Out"
                   ^  ^           |--(Exhausted)-------^
                   |  |           |
                   |  |        (Repair)
                   |  |           v
                   |  +-- ApplyPatch <-- RepairAgent

A loop in wfpy is an ordinary feedback edge with no backpressure, so it must
terminate by itself: Decide is the only exit. The gate checks more than stdout --
a kernel that compiles nothing still exits 0 -- so each attempt dumps IR and the
gate requires a real kernel plus all four IR levels. Structural change is
reported, not failed: fixing a tiling bug must change the tiling.

Run:
    ./run.sh repair <kernel.py>
    PYPTO_REPAIR_TRANSPORT=acp ./run.sh repair <kernel.py>   # real; spends tokens
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.repair_agent import RepairAgent  # noqa: E402
from src.lib.repair_config import BUDGET, CONNECTOR, IS_ACP, REPAIR_SKILL, TRANSPORT  # noqa: E402
from src.lib.runner import WF_OUT, default_run_mode  # noqa: E402
from src.tasks.apply_patch import ApplyPatch  # noqa: E402
from src.tasks.decide import Decide  # noqa: E402
from src.tasks.repair_report import RepairReport  # noqa: E402
from src.tasks.run_kernel import RunKernel  # noqa: E402
from wfpy import connect, run, workflow  # noqa: E402


@workflow(inputs={"Kernel": str}, outputs={"Out": str})
def repair_kernel() -> None:
    """Five actors, with the feedback edge closing back onto RunKernel.Retry."""
    runner = RunKernel(run_mode=default_run_mode())
    decide = Decide(budget=BUDGET)
    repair = RepairAgent()
    patch = ApplyPatch()
    report = RepairReport()

    connect("Kernel", runner.Initial)
    connect(runner.Result, decide.Result)

    connect(decide.Done, report.Done)
    connect(decide.Exhausted, report.Exhausted)

    # The loop: repair -> patch -> back into the runner's second input port.
    connect(decide.Repair, repair.Failure)
    connect(repair.Patched, patch.Patched)
    connect(patch.Kernel, runner.Retry)

    connect(report.Out, "Out")


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
    return 0 if any(ln.startswith("REPAIRED") for ln in lines) else 1


if __name__ == "__main__":
    sys.exit(main())
