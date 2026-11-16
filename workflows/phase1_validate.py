#!/usr/bin/env python3
# coding: utf-8
"""Phase 1 -- validate a PyPTO kernel as a wfpy dataflow graph.

Four nodes, one per phase, so each phase can later be swapped or fanned out
independently:

    "Kernel" --> Prepare --> RunSim --> Gate --(Ok)--> Report --> "Out"
                                         \\--(Bad)--/

- Prepare  resolves the kernel path and builds the command (no side effects).
- RunSim   runs the kernel in a subprocess, in its own work dir. This is a
           @task and not a @tool on purpose: a @tool raises on non-zero exit,
           which would abort the whole run instead of letting us route the
           failure to the Gate.
- Gate     routes on the result -- guarded action for success, unguarded
           fallback for failure. This is where a repair loop would attach.
- Report   renders a one-line verdict. Separate input ports for Ok and Bad,
           because a wfpy input port does not fan in.

Tokens are frozen dataclasses rather than dicts, deliberately. With a single
declared output port wfpy reinterprets a returned *dict* as a {port_name: value}
mapping (see wfpy/_action_runtime.py:269), so returning a dict payload emits
nothing at all, silently. A dataclass passes through untouched.

Run (after `source /usr/local/Ascend/ascend-toolkit/set_env.sh`):

    python3 workflows/phase1_validate.py
    wfpy run workflows/phase1_validate.py --workflow validate_kernel \
        --input Kernel=examples/01_fused_elementwise.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import subprocess
import sys
import time

from wfpy import Port, action, connect, guard, run, task, workflow

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KERNELS = [
    "examples/01_fused_elementwise.py",
    "examples/02_row_softmax.py",
    "examples/03_fused_linear.py",
]


def _first_error(stdout: str, stderr: str) -> str:
    """Pick the most informative failure line.

    The kernel's own ``FAIL:`` line is preferred: the simulator's helper scripts
    write unrelated warnings to stderr (a missing matplotlib, for instance), so
    the last stderr line is often not the actual cause.
    """
    for line in stdout.splitlines():
        if line.startswith("FAIL:"):
            return line[len("FAIL:"):].strip()
    for line in reversed(stderr.strip().splitlines()):
        if line.strip():
            return line.strip()
    return "no diagnostic output"


@dataclass(frozen=True)
class JobSpec:
    """What to run, and where."""

    kernel: str
    name: str
    exists: bool
    run_mode: str
    work_dir: str
    cmd: tuple[str, ...]


@dataclass(frozen=True)
class RunResult:
    """How the run went."""

    job: JobSpec
    rc: int
    stdout: str
    error: str
    seconds: float

    @property
    def passed(self) -> bool:
        # Both conditions matter: exit 0 alone would also accept a kernel that
        # printed nothing, and a PASS line alone would accept a late crash.
        return self.rc == 0 and "PASS:" in self.stdout


@task
class Prepare:
    """Resolve a kernel path into a job spec. Pure; no side effects."""

    # No default: in wfpy an annotated field without a default is a constructor
    # parameter, while one with a default becomes per-actor state instead.
    run_mode: str

    class Ports:
        Kernel = Port[str](direction="in")
        Job = Port[JobSpec](direction="out")

    @action(consumes={"Kernel": 1}, produces={"Job": 1})
    def prepare(self, kernel: str) -> JobSpec:
        path = Path(kernel)
        if not path.is_absolute():
            path = REPO_ROOT / path
        name = path.stem
        work_dir = REPO_ROOT / "runs" / f"{name}_{self.run_mode}"
        work_dir.mkdir(parents=True, exist_ok=True)
        return JobSpec(
            kernel=str(path),
            name=name,
            exists=path.is_file(),
            run_mode=self.run_mode,
            work_dir=str(work_dir),
            # sys.executable keeps us on the interpreter that has pypto + torch.
            cmd=(sys.executable, str(path), "--run_mode", self.run_mode),
        )


@task
class RunSim:
    """Run one kernel in a subprocess and capture its outcome."""

    timeout_s: int = 900  # has a default -> actor state, not a parameter

    class Ports:
        Job = Port[JobSpec](direction="in")
        Result = Port[RunResult](direction="out")

    @action(consumes={"Job": 1}, produces={"Result": 1})
    def execute(self, job: JobSpec) -> RunResult:
        if not job.exists:
            return RunResult(job=job, rc=-1, stdout="", error="kernel file not found", seconds=0.0)

        started = time.time()
        try:
            proc = subprocess.run(
                list(job.cmd),
                cwd=job.work_dir,
                env=os.environ.copy(),  # carries the CANN env + PYTHONPATH
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            rc, stdout = proc.returncode, proc.stdout
            error = "" if rc == 0 else _first_error(stdout, proc.stderr)
        except subprocess.TimeoutExpired:
            rc, stdout, error = -2, "", f"timeout after {self.timeout_s}s"

        return RunResult(
            job=job,
            rc=rc,
            stdout=stdout,
            error=error,
            seconds=round(time.time() - started, 1),
        )


@task
class Gate:
    """Route a result: Ok when the kernel ran and self-verified, else Bad."""

    class Ports:
        Result = Port[RunResult](direction="in")
        Ok = Port[RunResult](direction="out")
        Bad = Port[RunResult](direction="out")

    @action(consumes={"Result": 1}, produces={"Ok": 1})
    @guard(lambda self, result: result.passed)
    def accept(self, result: RunResult) -> RunResult:
        return result

    @action(consumes={"Result": 1}, produces={"Bad": 1})
    def reject(self, result: RunResult) -> RunResult:  # no guard -- the fallback
        return result


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
    prepare = Prepare(run_mode="sim")
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
