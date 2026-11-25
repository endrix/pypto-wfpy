"""Run the residual verifier on its assigned device and keep per-attempt evidence."""

import json
from pathlib import Path
from tempfile import mkdtemp

from src.lib.device_tune import device_lock
from src.lib.records import JobSpec
from src.lib.runner import PYTHON, REPO_ROOT, execute_job


class ResidualComparisonJob(JobSpec):
    def has_pass_marker(self, stdout):
        return "PASS: residual RMSNorm comparison" in stdout.splitlines()


def run_comparison(kernels, workspace, device):
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(dir=workspace, prefix="compare-"))
    manifest = directory / "candidates.json"
    manifest.write_text(json.dumps(kernels, indent=2))
    result = directory / "result.json"
    driver = REPO_ROOT / "tools/compare_residual.py"
    cmd = (PYTHON, "-u", str(driver), str(manifest), str(result))
    job = ResidualComparisonJob(str(driver), "residual_rmsnorm", True, "npu", str(directory),
                                cmd, device_id=device, phase="compare")
    with device_lock(device):
        run = execute_job(job, timeout_s=1800)
    if run.passed and result.is_file():
        return json.loads(result.read_text()), directory, ""
    return {}, directory, run.error or "verifier did not write a result"
