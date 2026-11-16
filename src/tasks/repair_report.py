#!/usr/bin/env python3
# coding: utf-8
"""Render the repair loop's verdict, including what the compiler produced."""

from __future__ import annotations

from src.records import Outcome
from src.repair_config import BUDGET
from wfpy import Port, action, task


@task
class RepairReport:
    """Done and Exhausted are separate ports: a wfpy input port does not fan in."""

    class Ports:
        Done = Port[Outcome](direction="in")
        Exhausted = Port[Outcome](direction="in")
        Out = Port[str](direction="out")

    @staticmethod
    def _graph_line(outcome: Outcome) -> str:
        g = outcome.graph
        return (f"      graph: compiled={g.compiled} levels={len(g.levels)}/4 "
                f"leaves={g.block_leaves} tiles={len(g.tile_shapes)}")

    @action(consumes={"Done": 1}, produces={"Out": 1})
    def on_done(self, outcome: Outcome) -> str:
        head = (f"REPAIRED  {outcome.result.job.name}: passes and compiled clean "
                f"({outcome.result.seconds}s, run {outcome.attempt})")
        return "\n".join([head, self._graph_line(outcome)])

    @action(consumes={"Exhausted": 1}, produces={"Out": 1})
    def on_exhausted(self, outcome: Outcome) -> str:
        head = (f"GAVE UP   after {BUDGET} repair attempt(s); last tried "
                f"{outcome.result.job.name}\n      reason: {outcome.why()}")
        return "\n".join([head, self._graph_line(outcome)])
