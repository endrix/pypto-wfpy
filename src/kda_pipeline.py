#!/usr/bin/env python3
# coding: utf-8
"""Author a PyPTO kernel from PyTorch, optimise it, redesign if that fails.

The top-level workflow:

    KernelSource --> combo_flow --+--> MarkdownViewer      the search report
                                  |--> MarkdownViewer      the advisor's analysis
                                  |--> SwimlaneViewer      the simulator trace
                                  |--> PyPTOGraphViewer x4 tensor/tile/block/execute
                                  |
                                  +--> "Out"

The whole arc in one workflow: a PyTorch module goes in, five gated authoring
stages turn it into a PyPTO kernel, the optimiser searches its tiling, and if
that search finds nothing a redesign agent writes the brief for a different
decomposition and authoring runs again.

Ten agents in the body. Three nested budgets, each varying a different thing:
the compiler-rejection repairs, the tiling rounds, and the designs.

The default subject is one stage of KDA -- Kimi Delta Attention's decayed Gram
matrix, lifted out of `kda_097/artifacts/kda_vectorized.py`. It is a fair target
in a way the smaller examples are not:

- two matmuls, an exponential and a triangular mask, so the tiling has real
  choices in it;
- (128, 64) with a (128, 128) result, which is large enough that occupancy is
  not automatically hopeless -- `04_layer_norm` at 8K elements ran on one core
  of 72 and no tiling could change that;
- and a numerical trap that a careless port fails: `exp(g[c]) * exp(-g[i])`
  overflows fp32 because `g` is a cumulative sum of log-space gates. The torch
  source demonstrates it -- 300 non-finite entries without the per-row-block
  reference exponent -- so a kernel that skips the blocking is caught rather
  than merely suspected.

Not the whole of KDA: the remaining stages need a recursive blocked triangular
inverse and a scan with a loop-carried dependence over chunks. pypto has `loop`,
`concat` and `assemble`, so both are expressible, but neither is a tile-shaped
problem and they would dominate the run.

Run:
    ./run.sh kda                                       # the KDA decay matrix
    ./run.sh kda torch_src/rms_norm.py rms_norm        # something smaller
    PYPTO_COMBO_REDESIGNS=0 ./run.sh kda               # author and optimise once
"""

from collections.abc import Sequence
import os
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib import cann  # noqa: E402
from src.lib.combo_config import (  # noqa: E402
    CONNECTOR,
    IS_ACP,
    MODEL,
    OUTER_BUDGET,
    TRANSPORT,
)
from src.lib.optimize_config import BUDGET, MIN_GAIN_PCT  # noqa: E402
from src.lib.run_options import agent_run_options, summary as kept_summary  # noqa: E402
from src.lib.runner import REPO_ROOT, WF_OUT  # noqa: E402
from src.sources.kernel_source import KernelSource  # noqa: E402
from src.viewers.markdown_viewer import MarkdownViewer  # noqa: E402
from src.viewers.pypto_graph_viewer import PyPTOGraphViewer  # noqa: E402
from src.viewers.swimlane_viewer import SwimlaneViewer  # noqa: E402
from src.workflows.combo_flow import combo_flow  # noqa: E402
from wfpy import File, connect, run, workflow  # noqa: E402

DEFAULT_SOURCE = "torch_src/kda_decay_matrix.py"


def _arg(env: str, index: int, default: str) -> str:
    """From the environment, else argv when this file is the program.

    Not argv alone: the operator name and workspace are constructor parameters,
    so the graph needs them at BUILD time, and whoever imports this module owns
    argv.
    """
    from_env = os.environ.get(env, "").strip()
    if from_env:
        return from_env
    if (
        __name__ == "__main__"
        and len(sys.argv) > index
        and not sys.argv[index].startswith("-")
    ):
        return sys.argv[index]
    return default


TORCH_SOURCE = _arg("PYPTO_TORCH_SOURCE", 1, DEFAULT_SOURCE)
if not Path(TORCH_SOURCE).is_absolute():
    TORCH_SOURCE = str(REPO_ROOT / TORCH_SOURCE)

OP = _arg("PYPTO_OP", 2, Path(TORCH_SOURCE).stem)
WORKSPACE = str(REPO_ROOT / "custom" / OP)


@workflow(inputs={}, outputs={"Out": File(ext=".md")})
def kda_pipeline() -> None:
    """The module, the loop, and seven viewers onto what the agents actually did.

    A report says what was decided; these say what happened. The swimlane is the
    simulator's own trace of the winning kernel, and the four graph viewers are
    the compiler's view of it at each IR level -- which is the only way to see
    whether a tiling change did what the agent claimed.

    An output port broadcasts, so the report reaches both a viewer and "Out".
    """
    source = KernelSource(path=File(TORCH_SOURCE))
    workspace = str(Path(WORKSPACE) / uuid4().hex)
    combo = combo_flow(
        OP, workspace, TORCH_SOURCE,
        transport=TRANSPORT, connector=CONNECTOR if IS_ACP else "", model=MODEL,
    )()

    report_view = MarkdownViewer()
    advice_view = MarkdownViewer()
    swim_view = SwimlaneViewer()
    # One viewer per IR level. The class holds no state, so an instance is just
    # "a place one graph can be shown".
    tensor_view, tile_view, block_view, execute_view = (PyPTOGraphViewer() for _ in range(4))

    connect(source.Out, combo.Source)

    connect(combo.Report, report_view.In)
    connect(combo.Report, "Out")

    connect(combo.Advice, advice_view.In)
    connect(combo.Swimlane, swim_view.Swimlane)
    connect(combo.TensorGraph, tensor_view.Graph)
    connect(combo.TileGraph, tile_view.Graph)
    connect(combo.BlockGraph, block_view.Graph)
    connect(combo.ExecuteGraph, execute_view.Graph)


def main() -> int:
    # Makes an IDE run behave like a run.sh run.
    cann.apply_to_process()
    if not Path(TORCH_SOURCE).is_file():
        print(f"[combo] no such PyTorch source: {TORCH_SOURCE}", file=sys.stderr)
        return 2

    rounds = OUTER_BUDGET + 1
    print(
        f"[combo] source={TORCH_SOURCE}\n"
        f"[combo] op={OP} workspace root={WORKSPACE}\n"
        f"[combo] budgets: designs={OUTER_BUDGET} (so up to {rounds} authoring "
        f"rounds), tiling rounds={BUDGET}, min gain={MIN_GAIN_PCT}%\n"
        f"[combo] keeping={kept_summary()}\n"
        f"[combo] transport={TRANSPORT} connector={CONNECTOR} model={MODEL if IS_ACP else '(n/a)'}",
        file=sys.stderr,
    )
    result = run(
        kda_pipeline,
        acp_connector=CONNECTOR if IS_ACP else None,
        out_dir=str(WF_OUT),
        **agent_run_options(),
    )
    require_npu = Path(TORCH_SOURCE).resolve() == REPO_ROOT / DEFAULT_SOURCE
    return report_status(result["Out"] or [], require_npu=require_npu)


def report_status(reports: Sequence, *, require_npu: bool) -> int:
    """Return success only for a completed search and its required device check."""
    success = bool(reports)
    for token in reports:
        path = Path(str(token))
        try:
            report = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            print(f"[combo] cannot read report {path}: {exc}", file=sys.stderr)
            success = False
            continue
        print("\n".join(report.splitlines()[:12]))
        title, _, body = report.partition("\n")
        verdict = body.lstrip().split("\n", 1)[0]
        completed = title == "# Optimization search" and verdict.startswith(
            ("**OPTIMIZED  best=", "**NO GAIN  best=")
        )
        if require_npu:
            section = body.partition("## Final NPU execution\n")[2].split("\n## ", 1)[0]
            completed = completed and section.lstrip().startswith("Passed all five correctness fixtures on NPU ") and (
                "\nNPU latency: median " in section
            )
        success = success and completed
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
