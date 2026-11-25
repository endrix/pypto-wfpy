#!/usr/bin/env python3
# coding: utf-8
"""The advisor agent: explains the measurement, outside the loop."""

from __future__ import annotations

from src.agents import english_output
from src.lib.optimize_config import (
    ADVISOR_PROMPT,
    ADVISOR_SKILL,
    CONNECTOR,
    IS_ACP,
    MODEL,
    TIMEOUT_MS,
    TRANSPORT,
)
from wfpy import File, Port, agent


def make_perf_advisor(
    *,
    transport: str = TRANSPORT,
    connector: str = CONNECTOR if IS_ACP else "",
    model: str = MODEL,
    cli_tools_mode: str = "wfpy-none",
) -> type:
    @agent(
        prompt=english_output(ADVISOR_PROMPT),
        transport=transport,
        cli_tools_mode=cli_tools_mode,
        connector=connector,
        # Named in the connector's vocabulary, not as an API id; see optimize_config.
        model=model,
        timeout_ms=TIMEOUT_MS,
        # cannbot's `perf-analyzer`, which reads exactly the directory this project
        # produces: it wants the `output_*` dir containing bubble_analysis.log, and
        # ships scripts/analyze_perf.py to parse it. None when it cannot be
        # resolved -- see optimize_config for the sub-skill naming problem.
        skill=ADVISOR_SKILL,
    )
    class PerfAdvisor:
        """Turn the numbers into a reading of them.

        Deliberately NOT in the optimization loop. It consumes the final
        measurement and produces prose, so a slow or expensive answer cannot stall
        the search, and a wrong answer cannot send the optimizer somewhere bad --
        it is a deliverable for a person, not an input to a decision.
        """

        class Ports:
            Analysis = Port[str](direction="in")
            Advice = Port[File](direction="out", ext=".md")

    return PerfAdvisor


PerfAdvisor = make_perf_advisor()
