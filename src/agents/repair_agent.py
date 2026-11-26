#!/usr/bin/env python3
# coding: utf-8
"""The repair agent: an LLM is just another actor."""

from __future__ import annotations

from src.agents import english_output
from src.lib.repair_config import CONNECTOR, IS_ACP, MODEL, PROMPT, REPAIR_SKILL, TIMEOUT_MS, TRANSPORT
from wfpy import File, Port, agent


# NOTE: `wfpy plan` REWRITES this decorator in place, flattening these constants
# into literals and appending its own camelCase defaults. That silently undoes
# the env-var wiring: an earlier commit shipped a hardcoded transport while
# the startup banner still printed the constant, so the banner claimed a
# transport the run was not using. After any `./run.sh plan` on a workflow
# that imports this, re-check the decorator below.
@agent(
    prompt=english_output(PROMPT),
    transport=TRANSPORT,
    connector=CONNECTOR if IS_ACP else "",
    model=MODEL,
    timeout_ms=TIMEOUT_MS,
    skill=REPAIR_SKILL,
)
class RepairAgent:
    class Ports:
        Failure = Port[str](direction="in")
        Patched = Port[File](direction="out", ext=".json")
