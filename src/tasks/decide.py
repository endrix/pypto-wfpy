#!/usr/bin/env python3
# coding: utf-8
"""Accept, ask for a repair, or give up. The repair loop's only exit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.lib.records import Outcome
from wfpy import Port, action, guard, task


@task
class Decide:
    """Three mutually exclusive guards, so exactly one action can ever fire.

    wfpy picks nondeterministically among actions that are all activated --
    declaration order happens to break the tie in this runtime, but the model
    does not promise it -- so each action states its full condition rather than
    relying on being declared after a guarded one.
    """

    budget: int

    _attempts: int = 0
    _baseline: Any = None  # GraphSummary of the first run, for the structural delta

    class Ports:
        Result = Port[Outcome](direction="in")
        Repair = Port[str](direction="out")
        Done = Port[Outcome](direction="out")
        Exhausted = Port[Outcome](direction="out")

    def _remember(self, outcome: Outcome) -> None:
        if self._baseline is None:
            self._baseline = outcome.graph

    @action(consumes={"Result": 1}, produces={"Done": 1})
    @guard(lambda self, outcome: outcome.ok)
    def accept(self, outcome: Outcome) -> Outcome:
        self._remember(outcome)
        return outcome

    @action(consumes={"Result": 1}, produces={"Repair": 1})
    @guard(lambda self, outcome: not outcome.ok and self._attempts < self.budget)
    def ask_repair(self, outcome: Outcome) -> str:
        self._remember(outcome)
        self._attempts += 1
        try:
            source = Path(outcome.result.job.kernel).read_text(encoding="utf-8")
        except OSError:
            source = "(source unavailable)"
        return (
            f"Attempt {self._attempts} of {self.budget}.\n"
            f"Kernel: {outcome.result.job.kernel}\n"
            f"Failure: {outcome.why()}\n\n"
            f"--- current source ---\n{source}"
        )

    @action(consumes={"Result": 1}, produces={"Exhausted": 1})
    @guard(lambda self, outcome: not outcome.ok and self._attempts >= self.budget)
    def give_up(self, outcome: Outcome) -> Outcome:
        self._remember(outcome)
        return outcome
