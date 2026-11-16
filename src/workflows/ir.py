#!/usr/bin/env python3
# coding: utf-8
"""Inspect how the PyPTO compiler rewrites the graph, pass by pass.

    "Kernel" --> PrepareIR --> RunWithIR --> CollectIR --> DiffIR --> "Report"

PyPTO lowers a kernel TENSOR_GRAPH -> TILE_GRAPH -> BLOCK_GRAPH -> EXECUTE_GRAPH
across ~39 passes, dumping a readable .tifwkgr graph before and after each one.
This pairs them up and diffs them.

Run:
    ./run.sh ir [kernel.py ...]
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.runner import WF_OUT, default_run_mode  # noqa: E402
from src.tasks.collect_ir import CollectIR  # noqa: E402
from src.tasks.diff_ir import DiffIR  # noqa: E402
from src.tasks.prepare_ir import PrepareIR  # noqa: E402
from src.tasks.run_with_ir import RunWithIR  # noqa: E402
from wfpy import connect, run, workflow  # noqa: E402

DEFAULT_KERNELS = ["examples/01_fused_elementwise.py"]


@workflow(inputs={"Kernel": str}, outputs={"Report": str})
def inspect_ir() -> None:
    """Wire the four IR phases together."""
    prepare = PrepareIR(run_mode=default_run_mode())
    runner = RunWithIR()
    collect = CollectIR()
    differ = DiffIR()

    connect("Kernel", prepare.Kernel)
    connect(prepare.Job, runner.Job)
    connect(runner.Run, collect.Run)
    connect(collect.Diffs, differ.Diffs)
    connect(differ.Report, "Report")


def main() -> int:
    kernels = sys.argv[1:] or DEFAULT_KERNELS
    failures = 0
    for kernel in kernels:
        result = run(inspect_ir, inputs={"Kernel": kernel}, out_dir=str(WF_OUT))
        for line in result["Report"] or ["(no output)"]:
            print(line)
            if line.startswith(("FAIL", "(no")):
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
