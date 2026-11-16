#!/usr/bin/env python3
# coding: utf-8
"""Route a run result on whether the kernel verified itself."""

from __future__ import annotations

from src.lib.records import RunResult
from wfpy import Port, action, guard, task


@task
class Gate:
    """Ok when the kernel ran and self-verified, else Bad.

    Both guards are stated in full rather than leaving ``reject`` as an unguarded
    fallback. With a fallback, a passing result activates *both* actions and only
    declaration order picks ``accept`` -- and wfpy's model says the choice among
    several activated actions is nondeterministic, with declaration order an
    implementation detail of this runtime rather than a promise
    (docs/dataflow-concepts.md, "Action selection").
    """

    class Ports:
        Result = Port[RunResult](direction="in")
        Ok = Port[RunResult](direction="out")
        Bad = Port[RunResult](direction="out")

    @action(consumes={"Result": 1}, produces={"Ok": 1})
    @guard(lambda self, result: result.passed)
    def accept(self, result: RunResult) -> RunResult:
        return result

    @action(consumes={"Result": 1}, produces={"Bad": 1})
    @guard(lambda self, result: not result.passed)
    def reject(self, result: RunResult) -> RunResult:
        return result
