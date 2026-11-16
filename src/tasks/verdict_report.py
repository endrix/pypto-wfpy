#!/usr/bin/env python3
# coding: utf-8
"""Render a one-line verdict for a validated kernel."""

from __future__ import annotations

from src.records import RunResult
from wfpy import Port, action, task


@task
class VerdictReport:
    """Ok and Bad are separate ports: a wfpy input port does not fan in."""

    class Ports:
        Ok = Port[RunResult](direction="in")
        Bad = Port[RunResult](direction="in")
        Out = Port[str](direction="out")

    @action(consumes={"Ok": 1}, produces={"Out": 1})
    def on_ok(self, result: RunResult) -> str:
        return f"PASS  {result.job.name:<22} {result.seconds}s  ({result.job.run_mode})"

    @action(consumes={"Bad": 1}, produces={"Out": 1})
    def on_bad(self, result: RunResult) -> str:
        return f"FAIL  {result.job.name:<22} rc={result.rc}  {result.error[:70]}"
