#!/usr/bin/env python3
# coding: utf-8
"""Resolve a kernel path into a job spec. Pure; no side effects."""

from __future__ import annotations

from src.lib import graph
from src.lib.records import JobSpec
from src.lib.runner import make_job
from wfpy import Port, action, task


@task
class Prepare:
    """Turn a kernel path into a JobSpec with its own work dir."""

    # No default on either: in wfpy an annotated field without a default is a
    # constructor parameter, while one with a default becomes per-actor state --
    # and both of these have to be set per instance.
    run_mode: str
    # Whether to dump the compiler's per-pass graphs. Off for a plain run, on
    # where something downstream will render the IR levels.
    dump_ir: bool

    class Ports:
        Kernel = Port[str](direction="in")
        Job = Port[JobSpec](direction="out")

    @action(consumes={"Kernel": 1}, produces={"Job": 1})
    def prepare(self, kernel: str) -> JobSpec:
        if not self.dump_ir:
            return make_job(kernel, self.run_mode, suffix=f"_{self.run_mode}")
        # Only the passes that sit on an IR level transition: 50 dump files
        # instead of roughly 230 for every pass.
        return make_job(
            kernel, self.run_mode, suffix=f"_{self.run_mode}",
            via_ir_runner=True, extra_args=("--passes", graph.DUMP_PASSES),
        )
