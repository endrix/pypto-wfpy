#!/usr/bin/env python3
# coding: utf-8
"""Repair-loop configuration. Plain values, no actors."""

from __future__ import annotations

import os
from pathlib import Path

BUDGET = int(os.environ.get("PYPTO_REPAIR_BUDGET", "2"))
TRANSPORT = os.environ.get("PYPTO_REPAIR_TRANSPORT", "mock")
CONNECTOR = os.environ.get("PYPTO_REPAIR_CONNECTOR", "claude")
IS_ACP = TRANSPORT == "acp"
# A full-file rewrite takes longer than wfpy's 2-minute default.
TIMEOUT_MS = int(os.environ.get("PYPTO_REPAIR_TIMEOUT_MS", "600000"))

# The cannbot pypto-* skills all target Tensor mode. wfpy resolves skill=<name>
# against WF_SKILLS_ROOT, its highest-priority root, which lets them be used in
# place; a symlink would be rejected by its containment check.
SKILLS_ROOT = Path(os.environ.get("WF_SKILLS_ROOT", "/home/endrix/git/pypto/cannbot-skills/ops"))
SKILL_NAME = os.environ.get("PYPTO_REPAIR_SKILL", "pypto-precision-debug")
# Only claim the skill if it is really there; wfpy raises if it cannot resolve.
REPAIR_SKILL = SKILL_NAME if (SKILLS_ROOT / SKILL_NAME / "SKILL.md").is_file() else None
if REPAIR_SKILL:
    os.environ.setdefault("WF_SKILLS_ROOT", str(SKILLS_ROOT))

PROMPT = """You are fixing a PyPTO (Ascend NPU) Tensor-mode kernel that fails
its own numeric self-check against a pure-torch reference.

Rules:
- Reply with the COMPLETE corrected Python file and nothing else. No prose, no fences.
- Change only what is needed to make the check pass. Keep the `--run_mode {sim,npu}`
  flag, the PASS:/FAIL: prints and the exit codes.
- Do not weaken the test: leave the tolerances, the torch reference and every
  assertion exactly as they are.
- `pypto.mean` does not exist in this version; use `pypto.sum` over the axis.
- Set vector tiles with `pypto.set_vec_tile_shapes` before vector ops and cube
  tiles with `pypto.set_cube_tile_shapes` before `pypto.matmul`.
- Reduce over the intended axis: `dim=-1` reduces rows, `dim=0` reduces columns.
- Tile shape is a performance choice, not a correctness one. PyPTO splits a
  reduction across tiles correctly even when a tile does not span the reduced
  axis, so do not "fix" a tile shape unless the failure is actually about it.
"""
