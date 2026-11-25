#!/usr/bin/env python3
# coding: utf-8
"""Configuration for authoring and then optimising in one loop.

The outer loop: author a kernel, measure it on the device, search for a faster
version, and -- if the search finds nothing -- ask for a different DESIGN rather
than a different tiling, then author again.

That is the distinction worth the extra loop. `optimize`'s inner loop varies the
tiling of a fixed design, which is catalogue items F-9, F-10 and S-2. When every
tiling of a design is rejected or slower, the lever is not another tiling; it is
the decomposition. cannbot's own flow allows this too -- its Stage 7 can send the
work back -- and nothing in this project did until now.

The outer budget is small on purpose. One redesign means a default run is
"author, optimise, and if that found nothing, reconsider once". Each round is
ten agents, one of them writing a whole kernel, so the cost is linear in this
number and visible in it.
"""

from __future__ import annotations

import os

OUTER_BUDGET = int(os.environ.get("PYPTO_COMBO_REDESIGNS", "1"))

TRANSPORT = os.environ.get("PYPTO_COMBO_TRANSPORT", "acp")
CONNECTOR = os.environ.get("PYPTO_COMBO_CONNECTOR", "codex")
IS_ACP = TRANSPORT == "acp"
MODEL = os.environ.get("PYPTO_COMBO_MODEL", "gpt-6.2-sol")
TIMEOUT_MS = int(os.environ.get("PYPTO_COMBO_TIMEOUT_MS", "1800000"))

REDESIGN_PROMPT = """Write the next authoring brief for the operator, original
PyTorch source, workspace and expected implementation named in the evidence.
Repeat those four identifiers in the brief. Read the current design and search
history in that workspace. Keep the operator signature, tolerances and fixtures.

If authoring, verification, or simulation failed, name the failing phase,
quote the diagnostic and propose a concrete correction and a check. If an
evaluation timed out before producing a score, say which measurement is missing.

If the search produced a usable score, describe the current decomposition,
the changes actually tried and their outcomes. Cite the case and component
that justify your next change. Describe the operations, tile shapes and
acceptance criterion. Quote hardware microseconds only when the evidence
contains a matching device profile. Keep those separate from model cycles.

Return the brief as Markdown.
"""
