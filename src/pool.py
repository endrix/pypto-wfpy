#!/usr/bin/env python3
# coding: utf-8
"""Validate several PyPTO kernels concurrently, in one graph.

    "Kernels" --> Dispatch --+--> Worker0 --+--> Join --> PoolSummary --> "Out"
                             +--> Worker1 --+
                             +--> Worker2 --+
                             +--> Worker3 --+

One actor never runs concurrently with itself, so real parallelism needs one
worker chain per slot. Two wfpy constraints shape the fan-out and fan-in, and
both are explained in src/tasks/dispatch.py and src/tasks/join.py.

Run:
    ./run.sh pool [kernel.py ...]        # PYPTO_POOL sets the slot count
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib import cann  # noqa: E402
from src.lib.runner import DEVICES, WF_OUT, default_run_mode  # noqa: E402
from src.tasks.dispatch import make_dispatch  # noqa: E402
from src.tasks.join import make_join  # noqa: E402
from src.tasks.pool_summary import PoolSummary  # noqa: E402
from src.tasks.worker import make_worker  # noqa: E402
from wfpy import connect, run, workflow  # noqa: E402

DEFAULT_KERNELS = [
    "examples/01_fused_elementwise.py",
    "examples/02_row_softmax.py",
    "examples/03_fused_linear.py",
    "examples/04_layer_norm.py",
    "examples/05_attention_block.py",
]
POOL = int(os.environ.get("PYPTO_POOL", "4"))

if POOL < 1:
    raise ValueError("PYPTO_POOL must be positive")
if default_run_mode() == "npu":
    POOL = min(POOL, len(DEVICES) or 1)

Dispatch = make_dispatch(POOL)
Join = make_join(POOL)
Workers = [make_worker(i) for i in range(POOL)]


@workflow(inputs={"Kernels": list}, outputs={"Out": str})
def validate_pool() -> None:
    """Fan the kernel list across POOL worker chains, then join."""
    dispatch = Dispatch()
    join = Join()
    summary = PoolSummary()
    workers = [Workers[i](run_mode=default_run_mode()) for i in range(POOL)]

    connect("Kernels", dispatch.Kernels)
    for i, worker in enumerate(workers):
        connect(getattr(dispatch, f"S{i}"), worker.Kernel)
        connect(worker.Result, getattr(join, f"R{i}"))
    connect(join.Results, summary.Results)
    connect(summary.Out, "Out")


def main() -> int:
    # Makes an IDE run behave like a run.sh run: run.sh sources
    # set_env.sh, a direct launch from an IDE does not.
    cann.apply_to_process()
    kernels = sys.argv[1:] or DEFAULT_KERNELS
    result = run(validate_pool, inputs={"Kernels": kernels}, max_workers=POOL,
                 out_dir=str(WF_OUT))
    lines = result["Out"] or []
    # The join emits a snapshot per arrival; the last one is the most complete.
    print(lines[-1] if lines else "(no output)")
    return 0 if lines and lines[-1].startswith(f"{len(kernels)}/{len(kernels)} kernels passed") else 1


if __name__ == "__main__":
    sys.exit(main())
