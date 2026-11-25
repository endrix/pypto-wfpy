#!/usr/bin/env python3
# coding: utf-8
"""Node: `stage_author` -- one authoring stage, built per skill."""

from __future__ import annotations

from typing import Callable

from src.agents import english_output
from src.lib.author_config import CONNECTOR, IS_ACP, MODEL, TIMEOUT_MS, TRANSPORT
from wfpy import File, Port, agent


def stage_author(
    stage: str, prompt: str, skill: str | None, ext: str, *,
    transport: str = TRANSPORT,
    connector: str = CONNECTOR if IS_ACP else "",
    model: str = MODEL,
) -> type:
    """Build the agent for one authoring stage.

    A factory and not five near-identical classes: the stages differ only in
    their skill, their prompt and the extension of the artifact they produce,
    and five copies of this file would drift. The one-actor-per-file rule is
    about actor definitions, and this file defines one -- parameterised, the way
    `src/tasks/dispatch.py` and `worker.py` already do for the pool.

    The output port declares its extension, so wfpy writes the reply to a file
    with that suffix and passes the path on. That is what lets a gate check the
    artifact and a viewer render it.
    """

    @agent(
        prompt=english_output(prompt),
        transport=transport,
        # native, not the wfpy-none default: these skills are written for an
        # agent that can read files -- templates, reference implementations, the
        # installed pypto package -- and with no tools a stage told to "read
        # SPEC.md" reads nothing and answers from the prompt alone. Measured
        # earlier in another agent's own words: "without tools (policy is
        # wfpy-none)".
        cli_tools_mode="native",
        connector=connector,
        model=model,
        timeout_ms=TIMEOUT_MS,
        skill=skill,
    )
    class StageAuthor:
        f"""The {stage} stage: reads the workspace, writes one artifact."""

        class Ports:
            # A File, not a string. wfpy stages a file input as a real file for
            # a CLI/ACP agent (`work/agent-inputs/<actor>/`, up to 200k chars)
            # instead of inlining it into the prompt, so a stage reads a
            # document rather than a prompt that happens to contain one.
            Brief = Port[File](direction="in")
            Artifact = Port[File](direction="out", ext=ext)

    StageAuthor.__name__ = f"{stage.capitalize()}Author"
    StageAuthor.__qualname__ = StageAuthor.__name__
    return StageAuthor
