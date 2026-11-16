#!/usr/bin/env python3
# coding: utf-8
"""Shared tokens and actors for the PyPTO wfpy workflows.

Tokens are frozen dataclasses rather than dicts on purpose. With a *single*
declared output port wfpy reinterprets a returned dict as a {port_name: value}
mapping (wfpy/_action_runtime.py:269), so returning a dict payload emits nothing
at all, silently. A dataclass passes through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from wfpy import Port, action, guard, task

REPO_ROOT = Path(__file__).resolve().parent.parent
IR_RUNNER = REPO_ROOT / "tools" / "run_with_ir.py"

# wfpy defaults out_dir to ./wf-out relative to the *invocation* cwd, which
# scatters run logs wherever the command happened to be typed. Pin it to the
# repo so every run's journal and queue trace lands in one place.
WF_OUT = Path(os.environ.get("PYPTO_WF_OUT", REPO_ROOT / "wf-out"))

# Join actors accumulate into state while several worker slots fire at once.
LOCK = threading.Lock()

# Device ids to spread concurrent slots over. One kernel per device: two runs
# sharing a device serialise on it, which would hide the parallelism.
DEVICES = [int(d) for d in os.environ.get("PYPTO_DEVICES", "0,1,2,3,4,5,6,7").split(",") if d.strip()]


def torch_npu_available() -> bool:
    """Whether the NPU backend is importable in this interpreter."""
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return True


def default_run_mode() -> str:
    """Run on the NPU when the backend is present, else fall back to the simulator.

    PYPTO_RUN_MODE forces either one. Sim mode is the Soc CAModel on CPU: it
    needs no device, but it is 100-1000x slower than hardware and its cycle
    counts are a model, not a measurement.
    """
    forced = os.environ.get("PYPTO_RUN_MODE", "").strip().lower()
    if forced in ("npu", "sim"):
        return forced
    return "npu" if torch_npu_available() else "sim"


def first_error(stdout: str, stderr: str) -> str:
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
    slot: int = 0
    device_id: int = 0


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

    @property
    def sim_output_dir(self) -> str:
        """Newest simulator output dir for this run, or "" if there is none."""
        outputs = [p for p in Path(self.job.work_dir).glob("output/*") if p.is_dir()]
        if not outputs:
            return ""
        newest = max(outputs, key=lambda p: p.stat().st_mtime)
        return str(newest)


def make_job(kernel: str, run_mode: str, suffix: str = "", slot: int = 0,
             via_ir_runner: bool = False) -> JobSpec:
    """Resolve a kernel path into a JobSpec with its own work dir."""
    path = Path(kernel)
    if not path.is_absolute():
        path = REPO_ROOT / path
    name = path.stem
    work_dir = REPO_ROOT / "runs" / f"{name}{suffix}"
    work_dir.mkdir(parents=True, exist_ok=True)
    if via_ir_runner:
        cmd = (sys.executable, str(IR_RUNNER), str(path), "--run_mode", run_mode)
    else:
        # sys.executable keeps us on the interpreter that has pypto + torch.
        cmd = (sys.executable, str(path), "--run_mode", run_mode)
    return JobSpec(
        kernel=str(path),
        name=name,
        exists=path.is_file(),
        run_mode=run_mode,
        work_dir=str(work_dir),
        cmd=cmd,
        slot=slot,
        device_id=DEVICES[slot % len(DEVICES)] if DEVICES else 0,
    )


def execute_job(job: JobSpec, timeout_s: int) -> RunResult:
    """Run one kernel in a subprocess and capture its outcome."""
    if not job.exists:
        return RunResult(job=job, rc=-1, stdout="", error="kernel file not found", seconds=0.0)

    env = os.environ.copy()  # carries the CANN env + PYTHONPATH
    # The kernels read TILE_FWK_DEVICE_ID; give each slot its own device so
    # concurrent runs do not serialise on one.
    env["TILE_FWK_DEVICE_ID"] = str(job.device_id)

    started = time.time()
    try:
        proc = subprocess.run(
            list(job.cmd),
            cwd=job.work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        rc, stdout = proc.returncode, proc.stdout
        if rc != 0:
            error = first_error(stdout, proc.stderr)
        elif "PASS:" not in stdout:
            # Exit 0 is not enough on its own: a file that runs but never
            # reaches its self-check (an empty or stubbed kernel, say) would
            # otherwise be reported as failing with no reason given.
            error = "exited 0 but printed no PASS: line"
        else:
            error = ""
    except subprocess.TimeoutExpired:
        rc, stdout, error = -2, "", f"timeout after {timeout_s}s"

    return RunResult(job=job, rc=rc, stdout=stdout, error=error,
                     seconds=round(time.time() - started, 1))


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
        return make_job(kernel, self.run_mode, suffix=f"_{self.run_mode}")


@task
class RunSim:
    """Run one kernel and report the outcome."""

    timeout_s: int = 900  # has a default -> actor state, not a parameter

    class Ports:
        Job = Port[JobSpec](direction="in")
        Result = Port[RunResult](direction="out")

    @action(consumes={"Job": 1}, produces={"Result": 1})
    def execute(self, job: JobSpec) -> RunResult:
        return execute_job(job, self.timeout_s)


@task
class Gate:
    """Route a result: Ok when the kernel ran and self-verified, else Bad.

    Both guards are stated in full rather than leaving ``reject`` as an unguarded
    fallback. With a fallback, a passing result activates *both* actions and only
    declaration order picks ``accept`` -- and wfpy's model says the choice among
    several activated actions is nondeterministic, with declaration order an
    implementation detail of this runtime rather than a promise
    (docs/dataflow-concepts.md, "Action selection"). Mutually exclusive guards
    mean exactly one action can ever fire.
    """

    class Ports:
        Result = Port[RunResult](direction="in")
        Ok = Port[RunResult](direction="out")
        Bad = Port[RunResult](direction="out")

    @action(consumes={"Result": 1}, produces={"Ok": 1})
    @guard(lambda self, result: result.passed)
    def accept(self, result: RunResult) -> RunResult:
        return result

    @action(consumes={"Result": 1}, produces={"Bad": 1})
    @guard(lambda self, result: not result.passed)
    def reject(self, result: RunResult) -> RunResult:
        return result
