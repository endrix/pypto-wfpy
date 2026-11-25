#!/usr/bin/env python3
# coding: utf-8
"""The repair agent for a candidate the COMPILER rejected."""

from __future__ import annotations

from src.agents import english_output
from src.lib.optimize_config import (
    CONNECTOR,
    DOCTOR_PROMPT,
    DOCTOR_SKILL,
    IS_ACP,
    MODEL,
    TIMEOUT_MS,
    TRANSPORT,
)
from wfpy import File, Port, agent


def make_tile_doctor(
    *,
    transport: str = TRANSPORT,
    connector: str = CONNECTOR if IS_ACP else "",
    model: str = MODEL,
    cli_tools_mode: str = "wfpy-none",
) -> type:
    @agent(
        prompt=english_output(DOCTOR_PROMPT),
        transport=transport,
        connector=connector,
        model=model,
        timeout_ms=TIMEOUT_MS,
        # `pypto-general-debug` routes tile-shape failures through its
        # DEBUG_GUIDEBOOK: L0A/L0B/L0C/L1 capacity, alignment, missing tile shape,
        # `set_cube_tile_shapes` validation. Exactly the INVALID_TILE class.
        skill=DOCTOR_SKILL,
        cli_tools_mode=cli_tools_mode,
    )
    class TileDoctor:
        """Make a rejected optimization legal instead of discarding it.

        Its own node rather than a branch of the applier because a compiler
        rejection and a performance loss are different events: one needs a
        constraint diagnosis, the other a different lever. Folding them together is
        what let a measured `INVALID_TILE` be scored as "infinitely slow" and
        forgotten, so the next round could repeat the same illegal tiling.
        """

        class Ports:
            Failure = Port[str](direction="in")
            Fixed = Port[File](direction="out", ext=".json")

    return TileDoctor


TileDoctor = make_tile_doctor()
