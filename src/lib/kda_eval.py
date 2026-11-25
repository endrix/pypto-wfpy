"""Isolated subprocesses for the KDA workflow's verification and cost-model cases."""

import os
from pathlib import Path
import shlex
import tempfile

from src.lib.records import JobSpec, RunResult
from src.lib.runner import DEVICES, PYTHON, REPO_ROOT, execute_job


def evaluate(phase: str, kernel: str, workspace: str, op: str) -> RunResult:
    root = Path(workspace) / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    directory = tempfile.mkdtemp(prefix=f"{Path(kernel).stem}-{phase}-", dir=root)
    cmd = (PYTHON, "-u", str(REPO_ROOT / "tools/check_kda.py"), phase, kernel,
           str(Path(workspace) / f"{op}_golden.py"))
    snapshot = root / "verified_golden.pt"
    generated = Path(directory) / "golden_outputs.pt"
    if phase == "golden":
        cmd += ("--golden-outputs", str(generated))
    elif phase in ("verify", "final"):
        cmd += ("--golden-outputs", str(snapshot))
    mode = "npu" if phase in ("golden", "verify", "final") else "sim"
    job = JobSpec(kernel, Path(kernel).stem, Path(kernel).is_file(), mode, directory, cmd,
                  device_id=DEVICES[0] if DEVICES else 0, phase=phase)
    limit = int(os.environ.get(f"PYPTO_KDA_{phase.upper()}_TIMEOUT_S", 900 if phase == "smoke" else 1800))
    if limit <= 0:
        raise ValueError(f"PYPTO_KDA_{phase.upper()}_TIMEOUT_S must be positive")
    print(f"[KDA {phase}] cwd={directory} timeout={limit}s {shlex.join(cmd)}", flush=True)
    result = execute_job(job, limit)
    if phase == "golden" and result.passed:
        # Publish only after all fixtures pass. Keep the per-attempt copy for the report.
        link = Path(directory) / "accepted.pt"
        os.link(generated, link)
        link.replace(snapshot)
    return result


def diagnostic(result: RunResult) -> str:
    fixture = next((line for line in reversed(result.stdout.splitlines()) if line.startswith("START ")),
                   "(driver did not start a fixture)")
    return (
        f"Last fixture: {fixture}\n"
        f"Phase: {result.job.phase}\nMode: {result.job.run_mode}\nDevice ID: {result.job.device_id}\n"
        f"Artifact: {result.job.kernel}\n"
        f"Command: {shlex.join(result.job.cmd)}\nWorking directory and logs: {result.job.work_dir}\n"
        f"Elapsed: {result.seconds}s\nExit: {result.rc}\n{result.error}\n"
        f"stdout (tail):\n{result.stdout[-4000:]}\nstderr (tail):\n{result.stderr[-4000:]}\n"
        "Simulator reports:\n" + ("\n".join(result.report_dirs) or "(none)")
    )
