#!/usr/bin/env python3
# coding: utf-8
"""Read the simulator's performance data and name the bottleneck.

    "Kernel" --> Prepare --> RunSim --> CollectPerf --> AnalyzePerf --> "Report"

Only works in sim mode: a device run writes compiled kernels and no performance
data, so this reports SKIP on hardware.

Run:
    ./run.sh perf [kernel.py ...]
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib import cann  # noqa: E402
from src.lib.runner import WF_OUT, default_run_mode  # noqa: E402
from src.tasks.analyze_perf import AnalyzePerf  # noqa: E402
from src.tasks.collect_perf import CollectPerf  # noqa: E402
from src.tasks.prepare import Prepare  # noqa: E402
from src.tasks.run_sim import RunSim  # noqa: E402
from wfpy import connect, run, workflow  # noqa: E402

DEFAULT_KERNELS = [
    "examples/01_fused_elementwise.py",
    "examples/02_row_softmax.py",
    "examples/03_fused_linear.py",
    "examples/04_layer_norm.py",
    "examples/05_attention_block.py",
]


@workflow(inputs={"Kernel": str}, outputs={"Report": str})
def analyze_perf() -> None:
    """Wire the four perf phases together."""
    prepare = Prepare(run_mode=default_run_mode(), dump_ir=False)
    runner = RunSim()
    collect = CollectPerf()
    analyze = AnalyzePerf()

    connect("Kernel", prepare.Kernel)
    connect(prepare.Job, runner.Job)
    connect(runner.Result, collect.Result)
    connect(collect.Perf, analyze.Perf)
    # analyze.Report (the .md file) is left unconnected: this entry point
    # prints to a terminal and has no viewer to hand a file to.
    connect(analyze.Summary, "Report")


def main() -> int:
    # Makes an IDE run behave like a run.sh run: run.sh sources
    # set_env.sh, a direct launch from an IDE does not.
    cann.apply_to_process()
    kernels = sys.argv[1:] or DEFAULT_KERNELS
    failures = 0
    for kernel in kernels:
        result = run(analyze_perf, inputs={"Kernel": kernel}, out_dir=str(WF_OUT))
        for line in result["Report"] or ["(no output)"]:
            print(line)
            if line.startswith(("SKIP", "(no")):
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
