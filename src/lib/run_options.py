#!/usr/bin/env python3
# coding: utf-8
"""Run options that decide how much of an agent run is kept.

An agent run throws away most of what makes it reviewable unless asked not to.
These are off by default in wfpy; they are on by default here, because a run
whose reasoning was not recorded cannot be checked afterwards -- and checking
afterwards is how several wrong conclusions in this project were caught.

What each one buys, from `wfpy.runner.run`:

- `agent_stream` -- streaming for agent calls, so a reply arrives progressively
  instead of appearing whole at the end. This is what makes a long call legible
  while it is still running.
- `agent_debug` -- writes the request, the response and the chat history to
  `agent-debug/` in the run directory. The durable record: the effective prompt
  actually sent (skill text included), and the raw reply before any port
  materialisation touched it.
- `keep_intermediates` -- copies the work directory into the run output, so the
  materialized port files survive instead of being cleaned up.

Set `PYPTO_AGENT_STREAM=0`, `PYPTO_AGENT_DEBUG=0` or `PYPTO_KEEP_INTERMEDIATES=0`
to turn one off for a run where the volume is not wanted.

Note what this cannot reach: these are arguments to `run()`, and wfpy has no
environment-variable override for them (`cli.py:348,357` read them from argparse
only). So they apply when this project's `main()` is the entry point -- through
run.sh, or an IDE launching the module directly. An IDE that instead shells out
to `wfpy run` bypasses `main()`, and the wfpy IDE extension exposes no setting
for these flags: it has `wfpy.agentTools*` and `wfpy.runOutputDir`, and nothing
for stream, debug, intermediates or verbose. Enabling them for that path needs
either an extension setting that passes run arguments through, or an environment
override in wfpy itself.
"""

from __future__ import annotations

import os


def _on(name: str) -> bool:
    return os.environ.get(name, "1") != "0"


AGENT_STREAM = _on("PYPTO_AGENT_STREAM")
AGENT_DEBUG = _on("PYPTO_AGENT_DEBUG")
KEEP_INTERMEDIATES = _on("PYPTO_KEEP_INTERMEDIATES")


def agent_run_options() -> dict:
    """The `run()` keyword arguments that keep an agent run reviewable."""
    return {
        "agent_stream": AGENT_STREAM,
        "agent_debug": AGENT_DEBUG,
        "keep_intermediates": KEEP_INTERMEDIATES,
    }


def summary() -> str:
    """One line for an entry point's banner, so a run says what it is keeping."""
    kept = [n for n, on in (("stream", AGENT_STREAM), ("debug", AGENT_DEBUG),
                            ("intermediates", KEEP_INTERMEDIATES)) if on]
    return "+".join(kept) if kept else "nothing"
