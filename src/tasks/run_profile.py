#!/usr/bin/env python3
# coding: utf-8
"""Node: `RunProfile` -- run an instrumented kernel and find its profile data."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import sys

from src.lib import profile
from src.lib.records import Measurement
from src.lib.runner import execute_job, make_job
from wfpy import Port, action, task

# Profiling copies per-task data off the device, so it needs a real one.
RUN_MODE = "npu"


@task
class RunProfile:
    """Run the kernel, locate the artifacts, and read the S-20 hint.

    The directory is found by its marker file, never by order: a run creates
    several `output_*` directories and only one holds the data, so taking the
    first or the newest by name gets an empty one.

    The S-20 hint (`Recommended: set max_workspace_kb near NNNN KB`) is parsed
    from stdout here because this is the only place that sees it. The
    orchestrator's S2 gate is not satisfied until it has been read and set in
    `runtime_options`, so carrying it forward is a declared step, not an extra.
    """

    label: str
    slot: int

    timeout_s: int = 1800

    class Ports:
        Instrumented = Port[str](direction="in")
        Profile = Port[Measurement](direction="out")

    @action(consumes={"Instrumented": 1}, produces={"Profile": 1})
    def run(self, instrumented: str) -> Measurement:
        job = make_job(instrumented, RUN_MODE, slot=self.slot,
                       work_dir=Path(instrumented).parent / "measurement")
        digest = sha256(Path(instrumented).read_bytes()).hexdigest()
        result = execute_job(job, self.timeout_s)
        profile_dir = profile.find_profile_dir(job.work_dir)
        workspace_kb = profile.recommended_workspace_kb(result.stdout)
        print(
            f"[profile] {self.label} (dev {job.device_id}): {job.name} -> "
            f"{'passed' if result.passed else 'failed its check'}; "
            f"data {'found' if profile_dir else 'MISSING'}"
            + (f"; recommends max_workspace_kb={workspace_kb}" if workspace_kb else ""),
            file=sys.stderr,
        )
        return Measurement(job=job, passed=result.passed, profile_dir=profile_dir,
                          workspace_kb=workspace_kb, error=result.error, instrumented_sha256=digest)
