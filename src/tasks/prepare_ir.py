#!/usr/bin/env python3
# coding: utf-8
"""Build the IR-dumping command for one kernel."""

from __future__ import annotations

from src.lib.records import JobSpec
from src.lib.runner import make_job
from wfpy import Port, action, task


@task
class PrepareIR:
    """Resolve a kernel into a job routed through tools/run_with_ir.py.

    Uses the shared make_job rather than rebuilding the path logic: the only
    difference from a plain run is via_ir_runner, which enables compile_debug_mode
    without editing the kernel.
    """

    run_mode: str

    class Ports:
        Kernel = Port[str](direction="in")
        Job = Port[JobSpec](direction="out")

    @action(consumes={"Kernel": 1}, produces={"Job": 1})
    def prepare(self, kernel: str) -> JobSpec:
        return make_job(kernel, self.run_mode, suffix="_ir", via_ir_runner=True)
