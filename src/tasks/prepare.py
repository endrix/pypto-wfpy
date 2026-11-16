#!/usr/bin/env python3
# coding: utf-8
"""Resolve a kernel path into a job spec. Pure; no side effects."""

from __future__ import annotations

from src.lib.records import JobSpec
from src.lib.runner import make_job
from wfpy import Port, action, task


@task
class Prepare:
    """Turn a kernel path into a JobSpec with its own work dir."""

    # No default: in wfpy an annotated field without a default is a constructor
    # parameter, while one with a default becomes per-actor state instead.
    run_mode: str

    class Ports:
        Kernel = Port[str](direction="in")
        Job = Port[JobSpec](direction="out")

    @action(consumes={"Kernel": 1}, produces={"Job": 1})
    def prepare(self, kernel: str) -> JobSpec:
        return make_job(kernel, self.run_mode, suffix=f"_{self.run_mode}")
