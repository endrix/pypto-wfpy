#!/usr/bin/env python3
# coding: utf-8
"""Paths, run-mode detection and subprocess execution.

Plain helpers, no actors. Everything here is deliberately location-independent:
`runs/` and `wf-out/` resolve against the repo, not the invocation cwd.
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import mkdtemp
from uuid import uuid4
import threading
import time

from src.lib import cann
from src.lib.records import JobSpec, RunResult

# src/lib/runner.py -> parents[0]=lib, [1]=src, [2]=project root.
REPO_ROOT = Path(__file__).resolve().parents[2]
IR_RUNNER = REPO_ROOT / "tools" / "run_with_ir.py"

# The interpreter the kernels run under. The project venv is preferred over
# sys.executable because an IDE may be running the workflow on a different
# interpreter that has wfpy but not pypto and torch.
_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.is_file() else sys.executable

# wfpy defaults out_dir to ./wf-out relative to the *invocation* cwd, which
# scatters run logs wherever the command happened to be typed. Pin it to the repo.
WF_OUT = Path(os.environ.get("PYPTO_WF_OUT", REPO_ROOT / "wf-out"))

# Join actors accumulate into state while several worker slots fire at once.
LOCK = threading.Lock()

# Device ids to spread concurrent slots over. One kernel per device: two runs
# sharing a device serialise on it, which would hide the parallelism.
DEVICES = [int(d) for d in os.environ.get("PYPTO_DEVICES", "0,1,2,3,4,5,6,7").split(",") if d.strip()]


_npu_probe: bool | None = None


def torch_npu_available() -> bool:
    """Whether the NPU backend works in the interpreter that runs the kernels.

    Decided from what is INSTALLED for the kernel interpreter, not by importing
    here, for two reasons. The
    kernels run as subprocesses under PYTHON with the CANN environment, so that
    is the interpreter whose capability matters. And an in-process import cannot
    succeed from a bare IDE launch anyway: cann.apply_to_process() can set
    LD_LIBRARY_PATH in os.environ, but the dynamic loader read it at process
    start, so torch_npu's shared objects would still fail to resolve and the run
    would silently fall back to sim.
    """
    global _npu_probe
    if _npu_probe is None:
        # A filesystem check, not a subprocess that imports torch. This runs on
        # every graph BUILD -- including `wfpy plan`, which the IDE calls
        # constantly -- and spawning an interpreter to import torch each time
        # costs seconds and makes the editor feel broken.
        # No .resolve(): a venv's bin/python is a symlink to the system
        # interpreter, so resolving it walks out of the venv and globs
        # /usr/lib/... where torch_npu is not installed.
        _npu_probe = any(
            (p / "torch_npu").is_dir()
            for p in Path(PYTHON).parent.parent.glob("lib/python*/site-packages")
        )
    return _npu_probe


def default_run_mode() -> str:
    """Run on the NPU when the backend is present, else fall back to the simulator.

    PYPTO_RUN_MODE forces either one. Sim mode is the Soc CAModel on CPU: it needs
    no device, but its cycle counts are a cost model, not a measurement.
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


def make_job(kernel: str, run_mode: str, suffix: str = "", slot: int = 0,
             via_ir_runner: bool = False, extra_args: tuple[str, ...] = (),
             work_dir: Path | None = None) -> JobSpec:
    """Resolve a kernel path into a JobSpec with its own work dir."""
    # str() rather than Path(kernel) directly: a @source emits its File
    # parameter as the dataclass, not as a resolved path string, so a kernel can
    # arrive here as File, str or Path. File.__str__ gives its path.
    path = Path(str(kernel))
    if not path.is_absolute():
        path = REPO_ROOT / path
    name = path.stem
    if work_dir is None:
        root = REPO_ROOT / "runs" / f"{name}{suffix}"
        root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(mkdtemp(prefix="run-", dir=root))
        pointer = root / f".latest-{uuid4().hex}"
        pointer.symlink_to(work_dir.name, target_is_directory=True)
        pointer.replace(root / "latest")
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
    if via_ir_runner:
        cmd = (PYTHON, str(IR_RUNNER), str(path), "--run_mode", run_mode)
    else:
        cmd = (PYTHON, str(path), "--run_mode", run_mode)
    return JobSpec(
        kernel=str(path),
        name=name,
        exists=path.is_file(),
        run_mode=run_mode,
        work_dir=str(work_dir),
        cmd=cmd + extra_args,
        slot=slot,
        device_id=DEVICES[slot % len(DEVICES)] if DEVICES else 0,
    )


def execute_job(job: JobSpec, timeout_s: int) -> RunResult:
    """Run one kernel in a subprocess and capture its outcome."""
    if not job.exists:
        return RunResult(job=job, rc=-1, stdout="", error="kernel file not found", seconds=0.0)

    # cann.environ(), not os.environ: an IDE launching a workflow directly has
    # never sourced set_env.sh, so pypto would not be importable in the child.
    env = cann.environ()
    # The kernels read TILE_FWK_DEVICE_ID; give each slot its own device so
    # concurrent runs do not serialise on one.
    env["TILE_FWK_DEVICE_ID"] = str(job.device_id)

    started = time.monotonic()
    timed_out = False
    # KDA cases have a private invocation directory. Persist both streams while
    # the job runs so a deadline or interrupted IDE run keeps the last fixture.
    if job.phase:
        directory = Path(job.work_dir)
        stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            with subprocess.Popen(job.cmd, cwd=directory, env=env, stdout=out, stderr=err,
                                  start_new_session=True) as proc:
                try:
                    rc = proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    rc, timed_out = -2, True
        stdout = stdout_path.read_text(errors="replace")
        stderr = stderr_path.read_text(errors="replace")
    else:
        try:
            proc = subprocess.run(job.cmd, cwd=job.work_dir, env=env, capture_output=True,
                                  text=True, timeout=timeout_s)
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            rc, timed_out = -2, True
            stdout = (exc.stdout or b"").decode(errors="replace")
            stderr = (exc.stderr or b"").decode(errors="replace")

    if timed_out:
        error = f"timeout after {timeout_s}s"
    elif rc != 0:
        error = first_error(stdout, stderr)
    elif not job.has_pass_marker(stdout):
        error = f"driver exited without completing KDA {job.phase}" if job.phase else "exited 0 but printed no PASS: line"
    else:
        error = ""
    reports = ()
    if job.phase:
        directories = Path(job.work_dir).glob("output/*/CostModelSimulationOutput")
        reports = tuple(str(p) for p in sorted(directories) if p.is_dir())
    return RunResult(job, rc, stdout, error, round(time.monotonic() - started, 1), stderr, reports)
