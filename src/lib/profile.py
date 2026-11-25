#!/usr/bin/env python3
# coding: utf-8
"""Collect and read what `debug_options={"runtime_debug_mode": 1}` produces.

This is the S2_COLLECT half of cannbot's tuning flow. A kernel run with that
option writes, into ONE directory:

    program.json              the compiled program
    merged_swimlane.json      per-task trace, in the shape the Toolkit renders
    bubble_analysis.log       per-core work and wait
    tilefwk_L1_prof_data.json per-core task spans
    dyn_topo.txt

and prints, on stdout, the S-20 hint the orchestrator's S2 gate requires:

    Recommended: set max_workspace_kb near 12288KB and above 96KB to activate
    memory-driven mode.

Two things measured here that a reader should not have to rediscover:

- **A run creates more than one `output_*` directory**, and the artifacts are in
  only one of them. Picking the first, or the newest by name, gets an empty one
  -- which is how the vendor script first came back with "No such file or
  directory: .../merged_swimlane.json" on a run that had produced it. Select the
  directory **by the artifact**, never by order.
- **This is measured time, not a cost model.** `analyze_swimlane.py` reports
  microseconds per subgraph from a device run, where the simulator reports
  cycles from a model. The two are not comparable, and only this one is a
  measurement.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

from src.lib import cann
from src.lib.runner import PYTHON
from src.lib.skill_paths import ops_root

# Where the tuning skill keeps the readers for this data. Used rather than
# reimplemented: they already know the swimlane's merge keys and hash order.
SCRIPTS = ops_root() / "pypto-op-perf-tune/tune-swimlane/scripts"
SWIMLANE_SCRIPT = SCRIPTS / "analyze_swimlane.py"
CORE_USAGE_SCRIPT = SCRIPTS / "analyze_core_usage.py"

# The artifact that marks the real output directory.
MARKER = "merged_swimlane.json"

# "Recommended: set max_workspace_kb near 12288KB and above 96KB to ..."
WORKSPACE_RE = re.compile(r"set max_workspace_kb near\s*(\d+)\s*KB", re.IGNORECASE)

DEBUG_OPTIONS = '{"runtime_debug_mode": 1}'
DEBUG_NOTE = "S2_COLLECT: swimlane and profiling data. The copy is the restore."


def find_profile_dir(work_dir: str | Path) -> str:
    """The output directory that actually holds the artifacts, or "".

    By marker, not by order: a run makes several `output_*` directories and only
    one has the data.
    """
    hits = sorted(
        Path(work_dir).glob(f"output/*/{MARKER}"),
        key=lambda p: p.stat().st_mtime,
    )
    return str(hits[-1].parent) if hits else ""


def recommended_workspace_kb(stdout: str) -> int:
    """The max_workspace_kb the run asked for, or 0 if it did not.

    The orchestrator's S2 gate is not satisfied until this has been read from
    stdout and set in `runtime_options`, so a flow that ignores it is skipping a
    declared step rather than making a different choice.
    """
    match = WORKSPACE_RE.search(stdout)
    return int(match.group(1)) if match else 0


def _run_script(script: Path, profile_dir: str, extra: tuple[str, ...] = ()) -> str:
    """Run one vendor reader over a profile directory and return its output."""
    if not script.is_file():
        return f"(missing script: {script})"
    if not profile_dir:
        return "(no profile directory: the run produced no swimlane data)"
    try:
        # stderr folded into stdout: these scripts write their whole REPORT to
        # stderr and still exit 0. Capturing stdout alone returns an empty
        # string from a script that worked perfectly -- which reads downstream
        # as "no data" rather than as a mistake in this wrapper.
        proc = subprocess.run(
            [PYTHON, str(script), profile_dir, *extra],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=300,
            # The project interpreter and the CANN environment, not bare
            # python3: analyze_core_usage.py asks torch.npu for the hardware's
            # core count and falls back to trace metadata without it, saying
            # "may underestimate" -- so the answer would be quietly wrong.
            env=cann.environ(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"({script.name} did not run: {exc})"
    if proc.returncode != 0:
        tail = proc.stdout.strip().splitlines()
        return f"({script.name} exited {proc.returncode}: {tail[-1] if tail else 'no output'})"
    return proc.stdout


def swimlane_report(profile_dir: str) -> str:
    """Per-subgraph timings and the merge-tuning guide, from the vendor reader."""
    return _run_script(SWIMLANE_SCRIPT, profile_dir)


def core_usage_report(profile_dir: str, device_id: int = 0) -> str:
    """Per-leafHash core usage against the hardware's core count."""
    return _run_script(CORE_USAGE_SCRIPT, profile_dir, ("--device-id", str(device_id)))
