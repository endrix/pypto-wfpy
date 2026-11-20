#!/usr/bin/env python3
# coding: utf-8
"""The repair agent: an LLM is just another actor."""

from __future__ import annotations

from src.lib.repair_config import CONNECTOR, IS_ACP, PROMPT, REPAIR_SKILL, TIMEOUT_MS, TRANSPORT
from wfpy import File, Port, agent


# NOTE: `wfpy plan` REWRITES this decorator in place, flattening these constants
# into literals and appending its own camelCase defaults. That silently undoes
# the env-var wiring: an earlier commit shipped a hardcoded transport="mock"
# while the startup banner still printed TRANSPORT, so PYPTO_REPAIR_TRANSPORT=acp
# claimed to use the agent and did not. After any `./run.sh plan` on a workflow
# that imports this, re-check the decorator below.
@agent(
    prompt=PROMPT,
    transport=TRANSPORT,
    connector=CONNECTOR if IS_ACP else "",
    timeout_ms=TIMEOUT_MS,
    skill=REPAIR_SKILL,  # None when the cannbot skills are not on disk
    # The mock reply is deliberately NOT a fix, so the loop runs to its budget
    # and termination stays observable offline.
    mock_outputs={"Patched": '{"diagnosis": "mock", "kernel": "# mock transport: no real repair was attempted\\n"}'},
)
class RepairAgent:
    class Ports:
        Failure = Port[str](direction="in")
        Patched = Port[File](direction="out", ext=".json")
