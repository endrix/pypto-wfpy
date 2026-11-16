#!/usr/bin/env python3
# coding: utf-8
"""Paths, run-mode detection and subprocess execution.

Plain helpers, no actors. Everything here is deliberately location-independent:
`runs/` and `wf-out/` resolve against the repo, not the invocation cwd.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from src.lib.records import JobSpec, RunResult

# src/lib/runner.py -> parents[0]=lib, [1]=src, [2]=project root.
REPO_ROOT = Path(__file__).resolve().parents[2]
IR_RUNNER = REPO_ROOT / "tools" / "run_with_ir.py"

# wfpy defaults out_dir to ./wf-out relative to the *invocation* cwd, which
# scatters run logs wherever the command happened to be typed. Pin it to the repo.
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
             via_ir_runner: bool = False, extra_args: tuple[str, ...] = ()) -> JobSpec:
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
        cmd=cmd + extra_args,
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
            list(job.cmd), cwd=job.work_dir, env=env,
            capture_output=True, text=True, timeout=timeout_s,
        )
        rc, stdout = proc.returncode, proc.stdout
        if rc != 0:
            error = first_error(stdout, proc.stderr)
        elif "PASS:" not in stdout:
            # Exit 0 is not enough on its own: a file that runs but never reaches
            # its self-check would otherwise be reported as failing with no reason.
            error = "exited 0 but printed no PASS: line"
        else:
            error = ""
    except subprocess.TimeoutExpired:
        rc, stdout, error = -2, "", f"timeout after {timeout_s}s"

    return RunResult(job=job, rc=rc, stdout=stdout, error=error,
                     seconds=round(time.time() - started, 1))
