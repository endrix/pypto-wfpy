#!/usr/bin/env python3
# coding: utf-8
"""Node: `MergeKernel` -- the head of a loop, where two edges become one."""

from __future__ import annotations

from wfpy import Port, action, task


@task
class MergeKernel:
    """Forward whichever kernel arrived: the first one, or the next candidate.

    This node exists purely because a wfpy input port does NOT fan in. The
    optimization loop feeds a sub-workflow (`simulate_flow`) whose boundary has
    a single `Kernel` input, so the initial kernel and the loop's feedback edge
    cannot both connect to it. Two input ports on an actor in front of it can.

    `RunKernel` in the repair loop solves the same problem by having `Initial`
    and `Retry` ports of its own. Here the work downstream is a whole
    sub-workflow rather than one actor, so the merge has to be its own node --
    which is the better shape anyway: the phases stay reusable and the loop's
    join is visible in the graph instead of hidden inside a task.
    """

    _n: int = 0

    class Ports:
        Initial = Port[str](direction="in")
        Retry = Port[str](direction="in")
        Out = Port[str](direction="out")

    def _forward(self, kernel: str) -> str:
        self._n += 1
        return str(kernel)

    @action(consumes={"Initial": 1}, produces={"Out": 1})
    def from_initial(self, kernel: str) -> str:
        return self._forward(kernel)

    @action(consumes={"Retry": 1}, produces={"Out": 1})
    def from_retry(self, kernel: str) -> str:
        return self._forward(kernel)
