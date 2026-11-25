#!/usr/bin/env python3
# coding: utf-8
"""Repair-loop configuration. Plain values, no actors."""

from __future__ import annotations

import os

from src.lib.skill_paths import ops_root

BUDGET = int(os.environ.get("PYPTO_REPAIR_BUDGET", "2"))
# The real transport, always. There is no mock: every agent here is
# answered by a model, and a run that did not call one is not a run of
# this workflow. PYPTO_REPAIR_TRANSPORT can name another ACP-capable
# transport, but not a stand-in.
TRANSPORT = os.environ.get("PYPTO_REPAIR_TRANSPORT", "acp")
CONNECTOR = os.environ.get("PYPTO_REPAIR_CONNECTOR", "claude")
IS_ACP = TRANSPORT == "acp"
# A full-file rewrite takes longer than wfpy's 2-minute default.
TIMEOUT_MS = int(os.environ.get("PYPTO_REPAIR_TIMEOUT_MS", "600000"))

# The cannbot pypto-* skills all target Tensor mode. wfpy resolves skill=<name>
# against WF_SKILLS_ROOT, its highest-priority root, which lets them be used in
# place; a symlink would be rejected by its containment check.
# The FIRST entry: WF_SKILLS_ROOT is an os.pathsep list, and another config
# module may already have appended to it -- optimize_config adds the tuning
# directory so its sub-skills are nameable. Without the split,
# Path("ops:ops/pypto-op-perf-tune") is one bogus directory and this skill
# silently resolves to None, so whether the repair agent gets its skill would
# depend on which config module was imported first.
SKILLS_ROOT = ops_root()
SKILL_NAME = os.environ.get("PYPTO_REPAIR_SKILL", "pypto-precision-debug")
# Only claim the skill if it is really there; wfpy raises if it cannot resolve.
REPAIR_SKILL = SKILL_NAME if (SKILLS_ROOT / SKILL_NAME / "SKILL.md").is_file() else None
if REPAIR_SKILL:
    os.environ.setdefault("WF_SKILLS_ROOT", str(SKILLS_ROOT))

PROMPT = """You are fixing a PyPTO (Ascend NPU) Tensor-mode kernel that fails
its own numeric self-check against a pure-torch reference.

Rules:
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

Output format: **a single JSON object** and nothing else. The skill this agent
loads does not state an output format, so the format is stated here. No prose
outside the object, and no fence around it.

    {
      "diagnosis": "one line on what was wrong",
      "kernel": "the complete corrected Python file"
    }

Put the whole file in `kernel` as one JSON string, newlines escaped as \n. The
structure is the contract: a reply asked for as "the file and nothing else"
comes back fenced often enough that every reader grew its own fence-stripping
branch.
"""
