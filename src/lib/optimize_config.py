#!/usr/bin/env python3
# coding: utf-8
"""Optimization-loop configuration. Plain values, no actors."""

from __future__ import annotations

import os
from pathlib import Path

BUDGET = int(os.environ.get("PYPTO_OPT_BUDGET", "2"))
# acp is the default: these workflows exist to drive a real agent, and a
# default that quietly does not is how a mock run gets mistaken for a real
# one. Set PYPTO_OPT_TRANSPORT=mock for tests and offline work --
# the mock reply is deliberately not a usable answer, so the reject path
# stays exercised. A default run SPENDS TOKENS; the banner says so.
TRANSPORT = os.environ.get("PYPTO_OPT_TRANSPORT", "acp")
CONNECTOR = os.environ.get("PYPTO_OPT_CONNECTOR", "claude")
IS_ACP = TRANSPORT == "acp"
TIMEOUT_MS = int(os.environ.get("PYPTO_OPT_TIMEOUT_MS", "600000"))

# The session model, named in the CONNECTOR's own vocabulary -- not as an API
# model id. wfpy applies it by asking the agent for its `model` config option
# and matching the value, then the name, then a prefix, case-insensitively
# (`acp_client.configure_session`). Claude Code offers `default, sonnet, haiku,
# opus`, so "opus" selects Opus; "claude-opus-5" matches nothing, and wfpy then
# logs a warning and leaves the session on its default -- silently, as far as
# the run output goes. Check the "Session model set to ..." line to be sure.
MODEL = os.environ.get("PYPTO_OPT_MODEL", "opus")

# A candidate must beat the incumbent by more than this to be adopted. The
# simulator is deterministic, so this is not noise margin -- it is a floor on
# what counts as worth keeping a rewrite for.
MIN_GAIN_PCT = float(os.environ.get("PYPTO_OPT_MIN_GAIN_PCT", "2.0"))

# A candidate the compiler REJECTS is a different event from a candidate that is
# merely slower, and it gets its own budget. Without one, a doctor that keeps
# returning invalid tiles would loop forever: a wfpy feedback edge has no
# backpressure, so every cycle needs a counter that can only go up.
REPAIR_BUDGET = int(os.environ.get("PYPTO_OPT_REPAIR_BUDGET", "2"))

SKILLS_ROOT = Path(os.environ.get("WF_SKILLS_ROOT", "/home/endrix/git/pypto/cannbot-skills/ops")
                   .split(os.pathsep)[0])


def _multi_root_wfpy() -> bool:
    """Whether the installed wfpy can read several roots from WF_SKILLS_ROOT.

    A capability probe, not a version check: the support landed in
    huawei-csl/wfpy#61 without a version bump, and `_explicit_skills_roots` is
    the function that PR adds. Probing matters because the fallback is not
    merely degraded -- on an older wfpy a two-path value makes the WHOLE
    variable a single bogus directory, which would stop `pypto-precision-debug`
    resolving and break the repair loop as a side effect.
    """
    try:
        from wfpy import _agent_prompt_runtime as runtime
    except ImportError:
        return False
    return hasattr(runtime, "_explicit_skills_roots")


MULTI_ROOT = _multi_root_wfpy()


def _ensure_root(path: Path) -> None:
    """Append a skills root to WF_SKILLS_ROOT, keeping what is already there.

    Not `setdefault`: run.sh exports WF_SKILLS_ROOT before the interpreter
    starts, so a default never applies and the second root silently went
    missing -- while the on-disk check below still said the skill was
    available. The two have to agree.
    """
    current = [p for p in os.environ.get("WF_SKILLS_ROOT", "").split(os.pathsep) if p]
    if str(path) not in current:
        current.append(str(path))
    os.environ["WF_SKILLS_ROOT"] = os.pathsep.join(current)

# The analysis skill is a SUB-skill: cannbot ships it at
# ops/pypto-op-perf-tune/perf-analyzer/. wfpy rejects a '/' in a skill name, so
# it is only nameable when its parent directory is itself a skills root -- which
# is why WF_SKILLS_ROOT below lists two paths. That needs wfpy with
# multi-root WF_SKILLS_ROOT support (huawei-csl/wfpy#61); on an older wfpy the
# second path is ignored and ADVISOR_SKILL resolves to None, which is handled.
TUNE_ROOT = SKILLS_ROOT / "pypto-op-perf-tune"
ADVISOR_SKILL_NAME = os.environ.get("PYPTO_OPT_ADVISOR_SKILL", "perf-analyzer")
ADVISOR_SKILL = (
    ADVISOR_SKILL_NAME
    if MULTI_ROOT and (TUNE_ROOT / ADVISOR_SKILL_NAME / "SKILL.md").is_file()
    else None
)

# cannbot's `tune-frontend`: code-level PyPTO tuning -- loop rewriting, TileShape
# setting, reshape and data-movement. That is exactly what this agent does, so it
# is the right skill, not the "no skill" this started with.
#
# The parent `pypto-op-perf-tune` is NOT the one to name: it is a dispatcher that
# opens with "性能调优必须通过编排器(tune-orchestrator)驱动" and only tells the
# reader which phase skill to load. `tune-frontend` is the phase itself. Its own
# gates describe the three-phase discipline the orchestrator enforces, not a
# refusal to work, and it carries the lever catalogue this loop needs.
#
# Like the advisor it is a SUB-skill, so it needs TUNE_ROOT as a skills root.
OPTIMIZER_SKILL_NAME = os.environ.get("PYPTO_OPT_SKILL", "tune-frontend").strip()
OPTIMIZER_SKILL = (
    OPTIMIZER_SKILL_NAME
    if MULTI_ROOT
    and OPTIMIZER_SKILL_NAME
    and (TUNE_ROOT / OPTIMIZER_SKILL_NAME / "SKILL.md").is_file()
    else None
)

if ADVISOR_SKILL or OPTIMIZER_SKILL:
    # The flat collection first, so a plain skill name keeps resolving there,
    # then the tuning collection the two sub-skills live in.
    _ensure_root(SKILLS_ROOT)
    _ensure_root(TUNE_ROOT)


# `pypto-general-debug` routes tile-shape failures -- L0A/L0B/L0C/L1 exceeded,
# alignment, missing tile shape, `set_cube_tile_shapes` validation -- through its
# DEBUG_GUIDEBOOK. A top-level skill, so it needs no extra root.
DOCTOR_SKILL_NAME = os.environ.get("PYPTO_OPT_DOCTOR_SKILL", "pypto-general-debug").strip()
DOCTOR_SKILL = (
    DOCTOR_SKILL_NAME
    if DOCTOR_SKILL_NAME and (SKILLS_ROOT / DOCTOR_SKILL_NAME / "SKILL.md").is_file()
    else None
)
if DOCTOR_SKILL:
    _ensure_root(SKILLS_ROOT)


# The required artifacts, in `tune-frontend`'s own order. The gate checks for
# these before an edit is allowed: "⛔ 禁止：未完成阶段A+阶段B的分析就直接进入
# 阶段C逐项优化". Matched case-insensitively against the plan text.
REQUIRED_ARTIFACTS = (
    ("stage A", ("stage a", "阶段a", "阶段 a", "global analysis", "全局分析")),
    ("stage B", ("stage b", "阶段b", "阶段 b", "local analysis", "局部分析")),
    ("ranked item list", ("ranked", "排序", "清单", "priority", "优先")),
)


PLANNER_PROMPT = """You are analysing a PyPTO (Ascend NPU) Tensor-mode kernel to
decide ONE change that will make it faster. You do not write code in this step.

Follow the skill's three-stage order and produce its artifacts, in this reply:

1. **Stage A - global analysis** over catalogue items F-1..F-10 and F-16..F-18:
   loop structure, constants, reshape, TileShape. Say which apply and which do not.
2. **Stage B - local analysis** over F-11..F-15: data-movement level.
3. **A ranked item list**: the applicable items in priority order, each with the
   evidence from the measurements below that makes it applicable.
4. **The one item you choose**, as a line of the exact form:
   `CHOSEN: F-<n> <short name>`
   followed by the concrete edit to make, naming the constants or calls to change
   and their new values.

Rules:
- Work from the catalogue (`shared/optimization_catalog.md`), not from intuition.
  Respect each item's "applies when" column and its prerequisites.
- This machine is **DAV_2201** (Ascend 910B2), not DAV_3510/A5. Every catalogue
  item gated on `npuarch == 'DAV_3510'` is unavailable here - mark those as not
  applicable and move on.
- The objective is the simulated cycle count in the target below. Occupancy,
  utilization and pipe shares are diagnostics that explain cycles; they are not
  the thing being minimised. A change that raises occupancy and raises cycles has
  failed.
- Honour the tile constraints stated under "constraints" below if any are given:
  a tiling the compiler rejects scores as infinitely slow.
- Markdown, no fences around the whole reply.

Output format: **markdown**. The skill this agent loads does not state an output
format, so the format is stated here. Reply with markdown only -- no JSON
envelope of your own, and no fence wrapping the whole reply.
"""


APPLIER_PROMPT = """You are applying ONE already-decided change to a PyPTO
(Ascend NPU) Tensor-mode kernel.

The directive below names the catalogue item and the edit. Make exactly that
edit and nothing else.


Rules:
- Do not weaken the test: leave the torch reference, the tolerances and every
  assertion exactly as they are. Keep the `--run_mode {sim,npu}` flag, the
  PASS:/FAIL: prints and the exit codes.
- Do not make changes the directive did not ask for, and do not change the
  mathematics.
- If the directive says the plan was INCOMPLETE, reply with the source
  unchanged. Do not invent a change of your own.
- Record what you did as a comment at the edit site, naming the item id.
- `pypto.mean` does not exist in this version; use `pypto.sum` over the axis.

Output format: **a single JSON object** and nothing else. The skill this agent
loads does not state an output format, so the format is stated here. No prose
outside the object, and no fence around it.

    {
      "item": "the catalogue id you applied, e.g. F-9",
      "kernel": "the complete rewritten Python file"
    }

Put the whole file in `kernel` as one JSON string, newlines escaped as \n. The
structure is the contract: a reply asked for as "the file and nothing else"
comes back fenced often enough that every reader grew its own fence-stripping
branch.
"""


DOCTOR_PROMPT = """A PyPTO (Ascend NPU) Tensor-mode kernel was rewritten for
performance and the compiler REJECTED it. Fix the rejection.

This is not a performance task. The previous edit was a deliberate optimization;
keep its intent and make it legal.


Rules:
- `INVALID_TILE` means the tile shapes violate a hardware constraint - an L0A /
  L0B / L0C or L1 capacity, an alignment rule, or a shape the op does not accept.
  Work out the real limit and pick the largest legal tiling that keeps the
  optimization's intent, rather than reverting to the original.
- Keep the torch reference, the tolerances, every assertion, the
  `--run_mode {sim,npu}` flag, the PASS:/FAIL: prints and the exit codes exactly
  as they are.
- Say what the constraint was in a comment at the edit site, so the next round
  does not repeat it.

Output format: **a single JSON object** and nothing else. The skill this agent
loads does not state an output format, so the format is stated here. No prose
outside the object, and no fence around it.

    {
      "constraint": "the hardware constraint that was violated",
      "kernel": "the complete corrected Python file"
    }

Put the whole file in `kernel` as one JSON string, newlines escaped as \n. The
structure is the contract: a reply asked for as "the file and nothing else"
comes back fenced often enough that every reader grew its own fence-stripping
branch.
"""


ADVISOR_PROMPT = """You are reading simulator performance data for a PyPTO
(Ascend NPU) Tensor-mode kernel and explaining what limits it.

You are given cost-model output: pipe usage percentages, per-core work and wait
times, and the tasks that waited longest to be scheduled. Write a short analysis:

1. Name the single binding constraint, and say which number says so.
2. Say what that number does NOT prove, so the reader does not over-read it.
3. Name the one lever most likely to move it, and the metric that would confirm it.

Rules:
- Markdown, under 300 words, no fences. Do not restate the tables.
- These are cost-model estimates on a CPU, not measurements on hardware.
  Do not express them as wall-clock time or as a speedup.
- A low fleet utilization with few cores used is an occupancy problem, not a
  memory-bandwidth problem, however high the MTE_IN percentage looks.

Output format: **markdown**. The skill this agent loads does not state an output
format, so the format is stated here. Reply with markdown only -- no JSON
envelope of your own, and no fence wrapping the whole reply.
"""

OPTIMIZER_PROMPT = """You are making a PyPTO (Ascend NPU) Tensor-mode kernel
faster without changing what it computes.

Reply with the COMPLETE rewritten Python file and nothing else. No prose, no fences.

Rules:
- Do not weaken the test: leave the torch reference, the tolerances and every
  assertion exactly as they are. Keep the `--run_mode {sim,npu}` flag, the
  PASS:/FAIL: prints and the exit codes. A faster kernel that fails its own
  check is worthless and will be rejected.
- Change the tiling and the work decomposition, not the mathematics.
- Set vector tiles with `pypto.set_vec_tile_shapes` before vector ops and cube
  tiles with `pypto.set_cube_tile_shapes` before `pypto.matmul`.
- Tile shape is a performance choice, not a correctness one: PyPTO splits a
  reduction across tiles correctly even when a tile does not span the reduced
  axis. So a tile is yours to change.
- `pypto.mean` does not exist in this version; use `pypto.sum` over the axis.

Pick the change from the skill's own optimization catalogue
(`shared/optimization_catalog.md`), not from intuition:

- Say which catalogue item you are applying, by its id (F-1..F-18 for
  code-level work, S-1..S-21 for swimlane-level). One item per reply.
- Respect its "applies when" column and its prerequisites. An item whose
  precondition does not hold in the data above is not a candidate.
- Prefer the higher-starred items among those that do apply.
- This machine is **DAV_2201** (Ascend 910B2), not DAV_3510/A5. Every catalogue
  item gated on `npuarch == 'DAV_3510'` is unavailable here -- skip them.
- Raising core occupancy (S-2, core filling by TileShape) is a real lever but
  not a free one: more, smaller tiles add per-tile cost, and on a small problem
  that loses more than the parallelism gains. Judge it against the cycle count
  in the target above, which is the objective -- not against occupancy, which is
  only a diagnostic.
"""
