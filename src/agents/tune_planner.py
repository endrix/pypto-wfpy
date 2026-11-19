#!/usr/bin/env python3
# coding: utf-8
"""The planning agent: decides WHICH change to make, and writes no code."""

from __future__ import annotations

from src.lib.optimize_config import (
    CONNECTOR,
    IS_ACP,
    MODEL,
    OPTIMIZER_SKILL,
    PLANNER_PROMPT,
    TIMEOUT_MS,
    TRANSPORT,
)
from wfpy import Port, agent


@agent(
    prompt=PLANNER_PROMPT,
    transport=TRANSPORT,
    connector=CONNECTOR if IS_ACP else "",
    model=MODEL,
    timeout_ms=TIMEOUT_MS,
    # cannbot's `tune-frontend`. Its contract is three ORDERED stages -- global
    # analysis, local analysis, then per-item optimization -- and it forbids
    # editing before the first two are done. A single agent asked for a finished
    # file therefore fights its own skill: measured, that produced two kernels
    # the compiler rejected. Splitting the phase into plan and apply is what
    # makes the node contract match the skill contract.
    skill=OPTIMIZER_SKILL,
    mock_outputs={
        "Plan": (
            "## Stage A - global analysis\n_mock transport: no analysis was performed._\n\n"
            "## Stage B - local analysis\n_mock._\n\n"
            "## Ranked item list\n_mock._\n\nCHOSEN: F-0 no change (mock)\n"
        )
    },
)
class TunePlanner:
    """Analysis and a single decision, as text a gate can check.

    The plan is markdown rather than a structured token because what the gate
    verifies is exactly what the skill declares it will produce -- named analysis
    stages and a ranked list -- and because the applier downstream needs the
    reasoning, not just the verdict.
    """

    class Ports:
        Request = Port[str](direction="in")
        Plan = Port[str](direction="out")
