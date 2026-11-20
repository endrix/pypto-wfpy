#!/usr/bin/env python3
# coding: utf-8
"""Triage configuration. Plain values, no actors."""

from __future__ import annotations

import os
from pathlib import Path

TRANSPORT = os.environ.get("PYPTO_TRIAGE_TRANSPORT", "acp")
CONNECTOR = os.environ.get("PYPTO_TRIAGE_CONNECTOR", "claude")
IS_ACP = TRANSPORT == "acp"
MODEL = os.environ.get("PYPTO_TRIAGE_MODEL", "opus")
TIMEOUT_MS = int(os.environ.get("PYPTO_TRIAGE_TIMEOUT_MS", "600000"))

# Pass verification copies a tensor off the device, so it needs a real one. In
# sim the run dies in setup_verify_data with RT_MEMCPY_FAILED. Not a preference.
RUN_MODE = "npu"

SKILLS_ROOT = Path(
    os.environ.get("WF_SKILLS_ROOT", "/home/endrix/git/pypto/cannbot-skills/ops")
    .split(os.pathsep)[0]
)
SKILL_NAME = os.environ.get("PYPTO_TRIAGE_SKILL", "pypto-precision-debug")
ANALYST_SKILL = SKILL_NAME if (SKILLS_ROOT / SKILL_NAME / "SKILL.md").is_file() else None

ANALYST_PROMPT = """You are told exactly which op in a PyPTO (Ascend NPU)
Tensor-mode kernel first produced a wrong value, and you explain why.

The localisation is already done and is not in question. Two builds of the same
kernel were run with pass verification on, which dumps every op's output tensor;
the ops were aligned by op magic and compared in execution order. The first op
whose output differs is named below, with the whole comparison after it.

Write a short analysis:

1. **What that op is**, in terms of the kernel's source. The dumped opcodes are
   tile-level primitives, so one source line becomes several -- a `pypto.softmax`
   call lowers to ROWMAX_SINGLE, SUB, EXP, ROWSUM_SINGLE, DIV. Say which source
   construct the named op belongs to, and quote the line.
2. **Why it diverges**, as a claim about the source and not about the numbers.
   The ops before it agree exactly, so the cause is at that line, not upstream.
3. **The one-line fix**, as the exact replacement text.
4. **What the later rows show**: the difference grows as it propagates, so say
   which later divergences are consequences rather than separate faults.

Rules:
- Markdown, under 350 words, no fences around the whole reply.
- Do not propose a tolerance change. The fault is in the kernel.
- Do not re-derive the localisation or second-guess it; the ops before the named
  one are bit-identical.

Output format: **markdown**. The skill this agent loads does not state an output
format, so the format is stated here. Reply with markdown only -- no JSON
envelope of your own, and no fence wrapping the whole reply.
"""
