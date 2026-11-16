#!/usr/bin/env python3
# coding: utf-8
"""Validate one PyPTO kernel: run it and gate on its own numeric self-check.

    "Kernel" --> Prepare --> RunSim --> Gate --(Ok)--> VerdictReport --> "Out"
                                         \\--(Bad)--/

For several kernels at once see pool.py.

Run:
    ./run.sh validate [kernel.py ...]
"""

from __future__ import annotations

from pathlib import Path
import sys

# wfpy loads a workflow by path, so put the project root on sys.path before the
# src.* imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.runner import WF_OUT, default_run_mode  # noqa: E402
from src.tasks.gate import Gate  # noqa: E402
from src.tasks.prepare import Prepare  # noqa: E402
from src.tasks.run_sim import RunSim  # noqa: E402
from src.tasks.verdict_report import VerdictReport  # noqa: E402
from wfpy import connect, run, workflow  # noqa: E402

DEFAULT_KERNELS = [
    "examples/01_fused_elementwise.py",
    "examples/02_row_softmax.py",
    "examples/03_fused_linear.py",
    "examples/04_layer_norm.py",
]


@workflow(inputs={"Kernel": str}, outputs={"Out": str})
def validate_kernel() -> None:
    """Wire the four phases together."""
    prepare = Prepare(run_mode=default_run_mode())
    runner = RunSim()
    gate = Gate()
    report = VerdictReport()

    connect("Kernel", prepare.Kernel)
    connect(prepare.Job, runner.Job)
    connect(runner.Result, gate.Result)
    connect(gate.Ok, report.Ok)
    connect(gate.Bad, report.Bad)
    connect(report.Out, "Out")


def main() -> int:
    kernels = sys.argv[1:] or DEFAULT_KERNELS
    failures = 0
    for kernel in kernels:
        result = run(validate_kernel, inputs={"Kernel": kernel}, out_dir=str(WF_OUT))
        for line in result["Out"] or ["(no output)"]:
            print(line)
            if line.startswith(("FAIL", "(no")):
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
