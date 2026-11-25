#!/usr/bin/env python3
# coding: utf-8
"""The redesign agent: decides on a different decomposition, not a different tile."""

from __future__ import annotations

from src.agents import english_output
from src.lib.author_config import DESIGN_SKILL
from src.lib.combo_config import (
    CONNECTOR,
    IS_ACP,
    MODEL,
    REDESIGN_PROMPT,
    TIMEOUT_MS,
    TRANSPORT,
)
from wfpy import File, Port, agent


def make_redesign_advisor(
    *,
    transport: str = TRANSPORT,
    connector: str = CONNECTOR if IS_ACP else "",
    model: str = MODEL,
    contract: str = "",
) -> type:
    @agent(
        prompt=english_output(REDESIGN_PROMPT + contract),
        transport=transport,
        connector=connector,
        model=model,
        timeout_ms=TIMEOUT_MS,
        # The design skill, not a tuning one. This agent is deciding a
        # decomposition, which is what pypto-op-design is for; the tuning skills
        # assume the decomposition is settled and vary the tiling within it.
        skill=DESIGN_SKILL,
        cli_tools_mode="native",
    )
    class RedesignAdvisor:
        """Turn a failed search into a brief for a different design.

        Outside both inner loops and the only node that can restart authoring. It
        reads what the optimiser tried and what the device measured, and writes the
        brief the next authoring run begins from -- so the loop carries a reasoned
        change rather than a retry.
        """

        class Ports:
            Evidence = Port[File](direction="in")
            Brief = Port[File](direction="out", ext=".md")

    return RedesignAdvisor


RedesignAdvisor = make_redesign_advisor()
