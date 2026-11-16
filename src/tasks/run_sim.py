#!/usr/bin/env python3
# coding: utf-8
"""Run one kernel and report the outcome."""

from __future__ import annotations

from src.records import JobSpec, RunResult
from src.runner import execute_job
from wfpy import Port, action, task


@task
class RunSim:
    """Execute a kernel in a subprocess.

    A @task and not a @tool on purpose: a @tool raises on non-zero exit, which
    would abort the whole graph instead of letting a gate route the failure.
    """

    timeout_s: int = 900  # has a default -> actor state, not a parameter

    class Ports:
        Job = Port[JobSpec](direction="in")
        Result = Port[RunResult](direction="out")

    @action(consumes={"Job": 1}, produces={"Result": 1})
    def execute(self, job: JobSpec) -> RunResult:
        return execute_job(job, self.timeout_s)
