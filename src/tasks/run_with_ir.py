#!/usr/bin/env python3
# coding: utf-8
"""Run a kernel with IR dumping on, then locate the dump root."""

from __future__ import annotations

from pathlib import Path

from src.records import IRRun, JobSpec
from src.runner import execute_job
from wfpy import Port, action, task


@task
class RunWithIR:
    """Execute the kernel, then find the computation_graph it produced."""

    timeout_s: int = 1800

    class Ports:
        Job = Port[JobSpec](direction="in")
        Run = Port[IRRun](direction="out")

    @action(consumes={"Job": 1}, produces={"Run": 1})
    def execute(self, job: JobSpec) -> IRRun:
        result = execute_job(job, self.timeout_s)

        # Each run stamps a fresh output/<timestamp>/ dir; take the newest one
        # that actually holds per-pass dumps.
        dump_root = ""
        candidates = [p for p in Path(job.work_dir).glob("output/*") if p.is_dir()]
        for cand in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
            if (cand / "computation_graph").is_dir():
                dump_root = str(cand / "computation_graph")
                break

        return IRRun(job=job, rc=result.rc, error=result.error,
                     seconds=result.seconds, dump_root=dump_root)
