#!/usr/bin/env python3
# coding: utf-8
"""Configuration for authoring a PyPTO kernel from PyTorch source.

cannbot's kernel-development flow as a sequence of stages, each with a skill and
a declared artifact. The orchestrator that normally drives it says of itself that
it "不负责具体优化建议或代码修改" -- it loads a phase skill, checks that skill's
declared artifacts, and advances. Here the graph does that, so there is no
orchestrator agent.

The artifacts are the contract, and they accumulate in one workspace:

    custom/<op>/SPEC.md              pypto-intent-understand
    custom/<op>/API_REPORT.md        pypto-api-explore
    custom/<op>/<op>_golden.py       pypto-golden-generate
    custom/<op>/DESIGN.md            pypto-op-design
    custom/<op>/<op>_impl.py         pypto-op-develop

Each stage is told where the workspace is and reads what the earlier stages
wrote, which is how the cannbot skills expect to work -- they open
`custom/<op>/SPEC.md` rather than being handed its text. So the tokens between
stages stay small and the workspace is the shared state.

One known weakness, stated rather than hidden: `$PYPTO_DEVKIT_DIR` is empty on
this machine, so `pypto-docs-search` -- which `pypto-api-explore` and
`pypto-op-develop` both depend on for API documentation -- can only fetch by
fixed paths. The real API is still introspectable from the installed package
(`import pypto; dir(pypto)`), so a stage is under-informed rather than blind,
and the prompts say to use that.
"""

from __future__ import annotations

import os

from src.lib.skill_paths import ops_root

# The real transport, always. There is no mock: every agent here is
# answered by a model, and a run that did not call one is not a run of
# this workflow. PYPTO_AUTHOR_TRANSPORT can name another ACP-capable
# transport, but not a stand-in.
TRANSPORT = os.environ.get("PYPTO_AUTHOR_TRANSPORT", "acp")
CONNECTOR = os.environ.get("PYPTO_AUTHOR_CONNECTOR", "claude")
IS_ACP = TRANSPORT == "acp"
MODEL = os.environ.get("PYPTO_AUTHOR_MODEL", "opus")
# Authoring a kernel is the longest single agent task here: a design or an
# implementation takes minutes, and one doctor call in the tuning loop was
# measured at four.
TIMEOUT_MS = int(os.environ.get("PYPTO_AUTHOR_TIMEOUT_MS", "1800000"))

SKILLS_ROOT = ops_root()


def _skill(name: str) -> str | None:
    """A skill name only if it is really on disk; wfpy raises otherwise."""
    return name if (SKILLS_ROOT / name / "SKILL.md").is_file() else None


INTENT_SKILL = _skill("pypto-intent-understand")
API_SKILL = _skill("pypto-api-explore")
GOLDEN_SKILL = _skill("pypto-golden-generate")
DESIGN_SKILL = _skill("pypto-op-design")
DEVELOP_SKILL = _skill("pypto-op-develop")


# What each stage must leave behind, in order. The gate after a stage checks for
# its file: the orchestrator's one real job, and the only thing that keeps a
# stage from proceeding on an analysis that was never written.
STAGES = (
    ("intent", "SPEC.md"),
    ("api", "API_REPORT.md"),
    ("golden", "{op}_golden.py"),
    ("design", "DESIGN.md"),
    ("develop", "{op}_impl.py"),
)

# Below this a file is a stub, not an artifact. An agent that replies "I have
# written SPEC.md" produces a short file and a passing gate otherwise.
MIN_ARTIFACT_BYTES = int(os.environ.get("PYPTO_AUTHOR_MIN_BYTES", "400"))

# How much of the previous artifact travels inline in the next stage's brief.
# Bounded because five accumulated artifacts would otherwise grow the prompt
# without limit; the whole workspace is on disk for a stage that wants more.
MAX_INLINE = int(os.environ.get("PYPTO_AUTHOR_MAX_INLINE", "12000"))

KDA_CONTRACT = """
The mathematical KDA formula in the source defines the numerical contract.
For each retained (c,i), compute sum_d a[c,d] * exp(g[c,d] - g[i,d]) * k[i,d].
Return finite FP32 values matching the direct FP64 oracle (atol=rtol=1e-3)
and exactly zero masked entries on all five workflow fixtures. This applies to
both the Torch golden and the kernel. On the inclusive (2,1) wide-step fixture
with g=[0,-1000], the retained [1,1] value is a[1,0]*k[1,0]. The blocked
example overflows there; use stable arithmetic and preserve the finite result.
Document this requirement in the specification and preserve it on every retry.
"""


_SHARED = """
The workspace is `{workspace}`. Earlier stages have written their artifacts
there; read them rather than assuming what they say. The PyTorch source being
ported is `{torch_source}`. Resolve every skill path beginning with
`custom/<op>/` inside this exact workspace, including supporting files in `eval/`.

This machine: Ascend 910B2 (DAV_2201), CANN 9.2.0, pypto 0.2.1 in Tensor mode.
PyPTO Pro is not available here, so write Tensor-mode pypto only.

The documentation cache `$PYPTO_DEVKIT_DIR` is EMPTY on this machine, so
`pypto-docs-search` cannot search it. Get the API from the installed package
instead -- `import pypto; dir(pypto)`, and read
`/usr/local/Ascend/cann-9.2.0-beta.2/python/site-packages/pypto/*.py` -- and say
when you could not confirm something rather than guessing at it.

Output format: a single file's contents, and nothing else. No preamble, no
commentary around it, and no fence wrapping the whole reply. The runtime writes
your reply to `{artifact}` verbatim.
"""

INTENT_PROMPT = """You are turning a PyTorch module into a structured PyPTO
operator specification.

Read the PyTorch source and write `SPEC.md`: the operator's name, the exact
mathematical formula it computes, every input and output with shape and dtype,
which axes are dynamic, and the numerical tolerance that should be required of
it.

Be concrete where the source is concrete and explicit where it is not. If the
source leaves something open -- a dtype, a shape bound -- say so in the document
rather than inventing a value, because a later stage will build to whatever this
says.
""" + _SHARED

API_PROMPT = """You are checking whether PyPTO can express the operator in
`SPEC.md`, before anyone writes it.

Write `API_REPORT.md`: for each operation the spec needs, the pypto call that
provides it, its constraints, and the tile-shape requirements it imposes. Name
anything the spec needs that pypto does NOT provide, and what to do instead.

Two facts worth having: `pypto.mean` does not exist in this version -- a mean is
`pypto.sum` over the axis divided by the axis length -- and tile shape is a
performance choice rather than a correctness one, since PyPTO splits a reduction
across tiles correctly even when a tile does not span the reduced axis.
""" + _SHARED

GOLDEN_PROMPT = """You are writing the reference implementation the kernel will
be checked against.

Write `{artifact}`: a pure-torch implementation of the operator in `SPEC.md`,
exporting a function named `{golden_name}`. It is the definition of
correct, so it must be obviously right rather than clever -- use torch's own
operators, not a hand-rolled equivalent.

Do not import pypto here. This file is what pypto is judged against, so it must
not depend on it.
""" + _SHARED

DESIGN_PROMPT = """You are designing the PyPTO implementation.

Write `DESIGN.md`: how the computation decomposes into tile operations, the tile
shapes for each stage and why, where `set_vec_tile_shapes` and
`set_cube_tile_shapes` must be called, and the order of operations. Say what the
cube unit does and what the vector unit does.

Work from `SPEC.md` and `API_REPORT.md`. Where the API report said something is
unavailable, design around it rather than through it.
""" + _SHARED

DEVELOP_PROMPT = """You are writing the PyPTO kernel.

Write `{artifact}`: a complete, runnable Python file implementing the design in
`DESIGN.md`, checked against the golden reference. Follow the structure of the
examples in this repository -- a `@pypto.frontend.jit` kernel, a `reference()`
function, and a `main()` taking `--run_mode {{npu,sim}}` that prints a line
starting `PASS:` on success and `FAIL:` on mismatch, and exits 0 or 1 to match.

Two things that will cost you the run if you get them wrong:

- Do NOT write `from __future__ import annotations`. It turns annotations into
  strings, and `pypto.frontend.jit` reads the real annotation objects to tell a
  tensor parameter from a scalar; with it, the first tensor is rejected as a
  "Non-tensor parameter".
- Set the tile shapes before the operations that need them, and set cube tiles
  per matmul when a kernel has more than one of different shapes.
""" + _SHARED
