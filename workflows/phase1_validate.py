#!/usr/bin/env python3
# coding: utf-8
"""Phase 1 -- validate one PyPTO kernel as a wfpy dataflow graph.

Four nodes, one per phase, so each phase can be swapped or replaced
independently:

    "Kernel" --> Prepare --> RunSim --> Gate --(Ok)--> Report --> "Out"
                                         \\--(Bad)--/

- Prepare  resolves the kernel path and builds the command (no side effects).
- RunSim   runs the kernel in a subprocess, in its own work dir. This is a
           @task and not a @tool on purpose: a @tool raises on non-zero exit,
           which would abort the whole run instead of letting us route the
           failure to the Gate.
- Gate     routes on the result -- guarded action for success, unguarded
           fallback for failure. See phase4_repair.py for a loop on Bad.
- Report   renders a one-line verdict. Separate input ports for Ok and Bad,
           because a wfpy input port does not fan in.

This runs one kernel per graph. For several kernels concurrently see
pool_validate.py.

Run (after `source /usr/local/Ascend/ascend-toolkit/set_env.sh`):

    ./run.sh validate
    wfpy run workflows/phase1_validate.py --workflow validate_kernel \
        --input Kernel=examples/01_fused_elementwise.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# wfpy loads this file by path; make sibling modules importable either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nodes import Gate, Prepare, RunResult, RunSim, default_run_mode  # noqa: E402
from wfpy import Port, action, connect, run, task, workflow  # noqa: E402

DEFAULT_KERNELS = [
    "examples/01_fused_elementwise.py",
    "examples/02_row_softmax.py",
    "examples/03_fused_linear.py",
]


@task
class Report:
    """Render a verdict. Ok and Bad are separate ports: input ports do not fan in."""

    class Ports:
        Ok = Port[RunResult](direction="in")
        Bad = Port[RunResult](direction="in")
        Out = Port[str](direction="out")

    @action(consumes={"Ok": 1}, produces={"Out": 1})
    def on_ok(self, result: RunResult) -> str:
        return f"PASS  {result.job.name:<22} {result.seconds}s  ({result.job.run_mode})"

    @action(consumes={"Bad": 1}, produces={"Out": 1})
    def on_bad(self, result: RunResult) -> str:
        return f"FAIL  {result.job.name:<22} rc={result.rc}  {result.error[:70]}"


@workflow(inputs={"Kernel": str}, outputs={"Out": str})
def validate_kernel() -> None:
    """Wire the four phases together."""
    prepare = Prepare(run_mode=default_run_mode())
    runner = RunSim()
    gate = Gate()
    report = Report()

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
        result = run(validate_kernel, inputs={"Kernel": kernel})
        for line in result["Out"] or ["(no output)"]:
            print(line)
            if line.startswith(("FAIL", "(no")):
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
