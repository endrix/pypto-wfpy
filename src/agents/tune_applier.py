#!/usr/bin/env python3
# coding: utf-8
"""The applying agent: makes one decided change, and decides nothing."""

from __future__ import annotations

from src.lib.optimize_config import (
    APPLIER_PROMPT,
    CONNECTOR,
    IS_ACP,
    MODEL,
    TIMEOUT_MS,
    TRANSPORT,
)
from wfpy import File, Port, agent


@agent(
    prompt=APPLIER_PROMPT,
    transport=TRANSPORT,
    connector=CONNECTOR if IS_ACP else "",
    model=MODEL,
    timeout_ms=TIMEOUT_MS,
    # No skill on purpose. The decision was made upstream with `tune-frontend`
    # loaded; this step only transcribes it into the file. Loading the skill
    # again would re-open the analysis the gate just finished checking, and the
    # skill's own rule is that analysis precedes editing, not that it repeats.
    skill=None,
    mock_outputs={"Candidate": '{"item": "F-0", "kernel": "# mock transport: no edit was applied\\n"}'},
)
class TuneApplier:
    """One decided edit, one complete file out. Not trusted: it is measured."""

    class Ports:
        Directive = Port[str](direction="in")
        # .json, not .py: the item id and the source travel together, and
        # wfpy serialises a returned dict for a .json port itself.
        Candidate = Port[File](direction="out", ext=".json")
