#!/usr/bin/env python3
# coding: utf-8
"""Node: `RunVerify` -- run an instrumented kernel and collect its op dumps."""

from __future__ import annotations

import sys

from src.lib import verify
from src.lib.records import DumpSet
from src.lib.runner import execute_job, make_job
from src.lib.triage_config import RUN_MODE
from wfpy import Port, action, task


@task
class RunVerify:
    """Run the kernel, then read what pass verification dumped.

    Run and collect in one actor: splitting them would add a node without adding
    a decision, since the dumps are only ever read from the run that wrote them.

    The run mode is pinned to npu and not configurable. Pass verification copies
    a tensor off the device, so in sim it dies in `setup_verify_data` with
    RT_MEMCPY_FAILED -- a device is a requirement here, not a preference.

    A failing kernel is the normal case: the whole point is to triage one. So a
    non-zero exit is reported in the token rather than raised, and the dumps are
    collected either way, because a kernel that fails its own check still dumped
    every op on the way there.
    """

    label: str
    # A parameter, not a counter: two verification runs fire at the same time,
    # and DEVICES[slot % len(DEVICES)] is what keeps them off one NPU.
    slot: int

    timeout_s: int = 1800

    class Ports:
        Instrumented = Port[str](direction="in")
        Dumps = Port[DumpSet](direction="out")

    @action(consumes={"Instrumented": 1}, produces={"Dumps": 1})
    def run(self, instrumented: str) -> DumpSet:
        job = make_job(instrumented, RUN_MODE, suffix=f"_verify_{self.label}",
                       slot=self.slot)
        result = execute_job(job, self.timeout_s)
        verify_dir = verify.find_verify_dir(job.work_dir)
        dumps = verify.read_dumps(verify_dir) if verify_dir else {}
        print(
            f"[triage] {self.label} (dev {job.device_id}): {job.name} -> "
            f"{'passed' if result.passed else 'failed its check'}, "
            f"{len(dumps)} op dump(s)",
            file=sys.stderr,
        )
        return DumpSet(job=job, passed=result.passed, verify_dir=verify_dir,
                       dumps=dumps, error=result.error)
