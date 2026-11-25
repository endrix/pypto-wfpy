#!/usr/bin/env python3
# coding: utf-8
"""The analyst agent: explains the op the comparison already named."""

from __future__ import annotations

from src.agents import english_output
from src.lib.triage_config import (
    ANALYST_PROMPT,
    ANALYST_SKILL,
    CONNECTOR,
    IS_ACP,
    MODEL,
    TIMEOUT_MS,
    TRANSPORT,
)
from wfpy import File, Port, agent


@agent(
    prompt=english_output(ANALYST_PROMPT),
    transport=TRANSPORT,
    connector=CONNECTOR if IS_ACP else "",
    model=MODEL,
    timeout_ms=TIMEOUT_MS,
    # cannbot's `pypto-precision-debug`: user-code-level precision triage, which
    # is what is left once the op is known.
    skill=ANALYST_SKILL,
)
class PrecisionAnalyst:
    """Given the culprit op, say which source line it is and how to fix it.

    The agent is downstream of the localisation, not part of it. `CompareDumps`
    names the op by arithmetic; what needs judgement is mapping a tile-level
    opcode back to a source construct -- one `pypto.softmax` lowers to
    ROWMAX_SINGLE, SUB, EXP, ROWSUM_SINGLE, DIV -- and that is this node's job.

    One input port carrying the whole brief. An agent fires only when every
    input port holds a token, so splitting the op table, the culprit and the
    source across three ports would need a join whose only job is
    concatenation; the upstream writer composes it instead.
    """

    class Ports:
        Brief = Port[str](direction="in")
        # .md, so the reply is a document a viewer can render. For a
        # Port[File] wfpy writes it to <actor>__Analysis__<n>.md and
        # passes that path on, not the text.
        Analysis = Port[File](direction="out", ext=".md")
