#!/usr/bin/env python3
# coding: utf-8
"""Node factory: `make_worker(slot)` -- one concurrent kernel runner."""

from __future__ import annotations

from typing import Any

from src.lib.records import RunResult
from src.lib.runner import execute_job, make_job
from wfpy import Port, action, task


def make_worker(slot: int) -> type:
    """Build one worker slot: resolve the kernel, then run it.

    One actor never runs concurrently with *itself*, so real parallelism needs
    one worker chain per slot. Prepare and RunSim are merged here because
    splitting them would add a node without adding a decision.
    """

    def work(self: Any, kernel: str) -> RunResult:
        job = make_job(kernel, self.run_mode, suffix=f"_{self.run_mode}", slot=slot)
        return execute_job(job, self.timeout_s)

    ns: dict[str, Any] = {
        "__doc__": f"Worker slot {slot}: prepare and run one kernel.",
        "Ports": type("Ports", (), {
            "Kernel": Port[str](direction="in"),
            "Result": Port[RunResult](direction="out"),
        }),
        "__annotations__": {"run_mode": str, "timeout_s": int},
        # run_mode has no default -> constructor parameter.
        # timeout_s has one     -> actor state.
        "timeout_s": 900,
        "work": action(consumes={"Kernel": 1}, produces={"Result": 1})(work),
    }
    return task(type(f"Worker{slot}", (), ns))
