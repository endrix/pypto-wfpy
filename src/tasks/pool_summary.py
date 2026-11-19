#!/usr/bin/env python3
# coding: utf-8
"""Summarize a concurrent pool run."""

from __future__ import annotations

from src.lib.runner import LOCK
from wfpy import Port, action, task


@task
class PoolSummary:
    """Report on the largest snapshot seen, and split pass from fail."""

    # State: the join emits a snapshot per arrival, so keep the fullest one.
    # A tuple, not a list: an annotated field with a default becomes per-actor
    # state, and a mutable default would be one object shared by every instance.
    # This code only ever rebinds it, so the bug was latent -- but a single
    # `.append()` here would have made two pool runs in one process share a list.
    _best: tuple = ()

    class Ports:
        Results = Port[list](direction="in")
        Out = Port[str](direction="out")

    @action(consumes={"Results": 1}, produces={"Out": 1})
    def summarize(self, results: list) -> str:
        with LOCK:
            if len(results) >= len(self._best):
                self._best = tuple(results)
            best = list(self._best)

        lines = []
        for r in sorted(best, key=lambda x: x.job.slot):
            where = f"slot {r.job.slot}, {r.job.run_mode}"
            if r.job.run_mode == "npu":
                where += f" dev{r.job.device_id}"
            if r.passed:
                lines.append(f"  PASS  {r.job.name:<22} {r.seconds}s  ({where})")
            else:
                lines.append(f"  FAIL  {r.job.name:<22} rc={r.rc}  {r.error[:60]}")
        passed = sum(1 for r in best if r.passed)
        wall = max((r.seconds for r in best), default=0.0)
        serial = round(sum(r.seconds for r in best), 1)
        head = (f"{passed}/{len(best)} kernels passed  "
                f"(slowest slot {wall}s vs {serial}s if run serially)")
        return "\n".join([head, *lines])
