#!/usr/bin/env python3
# coding: utf-8
"""Compile with IR dumping, run, and summarize the resulting graph."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

from src import graph
from src.records import Outcome
from src.runner import execute_job, make_job
from wfpy import Port, action, task


@task
class RunKernel:
    """Resolve, run and summarize one attempt.

    Prepare and RunSim are one actor here: splitting them added a node without
    adding a decision. Two input ports because a wfpy input port does not fan in,
    so the initial kernel and the repair feedback edge cannot share one.
    """

    run_mode: str

    timeout_s: int = 900
    _n: int = 0

    class Ports:
        Initial = Port[str](direction="in")
        Retry = Port[str](direction="in")
        Result = Port[Outcome](direction="out")

    def _run(self, kernel: str) -> Outcome:
        self._n += 1
        job = make_job(kernel, self.run_mode, suffix=f"_repair{self._n}",
                       slot=self._n - 1, via_ir_runner=True,
                       # Only the four passes that expose the IR level
                       # transitions, not all 39: 50 files instead of ~230.
                       extra_args=("--passes", graph.DUMP_PASSES))
        result = execute_job(job, self.timeout_s)
        summary = graph.summarize(job.work_dir)
        outcome = Outcome(result=result, graph=summary, attempt=self._n)
        # "run", not "attempt": this counts kernel runs, while Decide counts
        # repair requests. A budget of N means N repairs and so N+1 runs.
        print(
            f"[repair] run {self._n}: {Path(kernel).stem} -> "
            f"{'ok' if outcome.ok else outcome.why()} "
            f"[compiled={summary.compiled} leaves={summary.block_leaves}]",
            file=sys.stderr,
        )
        return outcome

    @action(consumes={"Initial": 1}, produces={"Result": 1})
    def from_initial(self, kernel: str) -> Outcome:
        return self._run(kernel)

    @action(consumes={"Retry": 1}, produces={"Result": 1})
    def from_retry(self, kernel: str) -> Outcome:
        return self._run(kernel)
