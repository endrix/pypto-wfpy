"""Isolated device measurements and the paired-latency acceptance rule."""

from contextlib import contextmanager, nullcontext
import fcntl
import json
import math
import os
import re
from pathlib import Path
from statistics import median
from tempfile import gettempdir

from src.lib import profile
from src.lib.records import JobSpec
from src.lib.runner import PYTHON, REPO_ROOT, execute_job


@contextmanager
def device_lock(device: int):
    """Serialize experiments sharing this device among this user's workflow runs."""
    path = Path(gettempdir()) / f"pypto-wfpy-{os.getuid()}-device-{device}.lock"
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class DeviceTuneJob(JobSpec):
    def has_pass_marker(self, stdout: str) -> bool:
        return f"PASS: device tune {self.phase}" in stdout.splitlines()


def measure(phase, adapter, device, directory, *kernels):
    directory.mkdir(parents=True)
    cmd = (PYTHON, "-u", str(REPO_ROOT / "tools/device_tune_eval.py"), phase, adapter,
           *(str(path.resolve()) for path in kernels))
    mode = "sim" if phase == "smoke" else "npu"
    job = DeviceTuneJob(str(kernels[0]), kernels[0].stem, all(p.is_file() for p in kernels),
                        mode, str(directory), cmd, device_id=device, phase=phase)
    timeout = int(os.environ.get("PYPTO_DEVICE_TUNE_TIMEOUT_S", "1800"))
    if timeout <= 0:
        raise ValueError("PYPTO_DEVICE_TUNE_TIMEOUT_S must be positive")
    with device_lock(device) if mode == "npu" else nullcontext():
        result = execute_job(job, timeout)
    if not result.passed:
        detail = result.stderr.strip().splitlines()[-3:]
        raise ValueError(f"{phase} failed: {result.error}. {' '.join(detail)}\nLogs: {directory}")
    return directory


def device_profile(adapter, device, directory, kernel):
    measure("profile", adapter, device, directory, kernel)
    traces = sorted(directory.glob(f"output/*/{profile.MARKER}"))
    if not traces:
        raise ValueError(f"Device profile produced no swimlane. Logs: {directory}")
    lines = [f"# Device profile of {kernel.name}", f"Device {device}. Source: `{kernel}`."]
    tasks = {}
    for trace in traces:
        events = json.loads(trace.read_bytes())["traceEvents"]
        for event in events:
            if event.get("ph") != "X" or not event.get("args", {}).get("event-hint", "").startswith("Task:"):
                continue
            # Dispatch IDs change between runs. leafHash identifies the compiled task family.
            hint = event["args"]["event-hint"]
            leaf = re.search(r"leafHash:(\d+)", hint)
            name = f"leafHash:{leaf[1]}" if leaf else str(event["name"])
            tasks[name] = max(tasks.get(name, 0), float(event["dur"]))
        with device_lock(device):
            # The vendor core-usage reader opens the device to query its core count.
            core_report = profile.core_usage_report(str(trace.parent), device_id=device)
        lines += [f"\nTrace: `{trace}`", profile.swimlane_report(str(trace.parent)), core_report]
    longest = sorted(tasks, key=tasks.get, reverse=True)[:10]
    lines += ["\n## Longest individual tasks",
              *(f"- `{name}`: {tasks[name]:.3f} µs" for name in longest)]
    report = directory / "profile.md"
    report.write_text("\n".join(lines) + "\n")
    return report, traces[0], longest


def paired_verdict(samples):
    """Require a 1% median paired gain and wins in at least nine of ten pairs."""
    if len(samples) != 2 or any(len(group) != 10 for group in samples):
        raise ValueError("expected ten paired NPU samples")
    if not all(math.isfinite(x) and x > 0 for group in samples for x in group):
        raise ValueError("invalid NPU latency")
    baseline, candidate = samples
    ratios = [c / b for b, c in zip(baseline, candidate)]
    gain = 100 * (1 - median(ratios))
    wins = sum(r < 1 for r in ratios)
    reason = (f"{median(baseline):.3f} → {median(candidate):.3f} ms; median paired "
              f"gain {gain:.2f}%, {wins}/10 wins")
    return gain >= 1 and wins >= 9, reason


def compare_kda_pair(incumbent: Path, challenger: Path, directory: Path, device: int):
    """Time two independently verified, uninstrumented KDA implementations."""
    result = measure("paired", "src.lib.device_tune_kda", device, directory, incumbent, challenger)
    adopted, evidence = paired_verdict(json.loads((result / "pairs.json").read_text()))
    return adopted, f"{evidence}. Raw samples: `{result / 'pairs.json'}`."
