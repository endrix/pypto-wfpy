#!/usr/bin/env python3
# coding: utf-8
"""Search for a faster version of a PyPTO kernel, then explain the result.

Four nodes, two of which are sub-workflows and one a viewer:

    KernelSource --> optimize_flow --+--> MarkdownViewer
                                     |
                                     +--> "Out"

``optimize_flow`` nests ``simulate_flow`` and ``analyze_flow`` -- the same two
phases ``sim_analyze`` uses to report -- inside a bounded agent loop that
measures, asks for a rewrite, re-measures and keeps the winner.

Every run calls a model. There is no stand-in transport: the loop exists to
find out what an agent does to a kernel, and a run that answered itself would
only tell us the wiring holds -- which `wfpy plan` and the linter already do,
for nothing.

Run:
    ./run.sh optimize                                   # spends tokens
    ./run.sh optimize examples/04_layer_norm.py
    PYPTO_OPT_BUDGET=4 ./run.sh optimize                # more rounds
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging  # noqa: E402

from src.lib import agent_io, cann  # noqa: E402
from src.lib.run_options import agent_run_options, summary as kept_summary  # noqa: E402
from src.lib.optimize_config import (  # noqa: E402
    ADVISOR_SKILL,
    BUDGET,
    CONNECTOR,
    IS_ACP,
    MIN_GAIN_PCT,
    MODEL,
    OPTIMIZER_SKILL,
    TRANSPORT,
)
from src.lib.runner import REPO_ROOT, WF_OUT  # noqa: E402
from src.sources.kernel_source import KernelSource  # noqa: E402
from src.viewers.markdown_viewer import MarkdownViewer  # noqa: E402
from src.workflows.optimize_flow import optimize_flow  # noqa: E402
from wfpy import File, connect, run, workflow  # noqa: E402

DEFAULT_KERNEL = "examples/01_fused_elementwise.py"


def _requested_kernel() -> str:
    """Which kernel the source node should hold.

    PYPTO_KERNEL, else argv[1] when this file is the program. Not argv[1]
    unconditionally: the source's path is a constructor parameter, so the graph
    needs it at BUILD time, and whoever imports this module owns argv.
    """
    from_env = os.environ.get("PYPTO_KERNEL", "").strip()
    if from_env:
        return from_env
    if __name__ == "__main__" and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]
    return DEFAULT_KERNEL


_requested = _requested_kernel()
KERNEL_PATH = _requested if Path(_requested).is_absolute() else str(REPO_ROOT / _requested)


@workflow(inputs={}, outputs={"Out": File(ext=".md")})
def optimize_kernel() -> None:
    """Three nodes: where the kernel comes from, the search, and the report."""
    kernel = KernelSource(path=File(KERNEL_PATH))
    search = optimize_flow()
    report = MarkdownViewer()
    advice = MarkdownViewer()

    connect(kernel.Out, search.Kernel)
    connect(search.Report, report.In)
    connect(search.Advice, advice.In)
    connect(search.Advice, "Out")


def main() -> int:
    # Makes an IDE run behave like a run.sh run: run.sh sources set_env.sh, a
    # direct launch from an IDE does not.
    cann.apply_to_process()
    if IS_ACP:
        # wfpy reports the model the session actually accepted at INFO
        # ("Session model set to ..."), and a model the connector does not offer
        # is a WARNING it otherwise swallows. Without this the run looks
        # identical whether `opus` was applied or quietly ignored.
        logging.basicConfig(level=logging.INFO, format="[wfpy:%(levelname)s] %(message)s")
        logging.getLogger("wfpy").setLevel(logging.INFO)
    print(
        f"[opt] source={KERNEL_PATH}\n"
        f"[opt] keeping={kept_summary()}\n"
        f"[opt] transport={TRANSPORT} connector={CONNECTOR if IS_ACP else '(n/a)'} "
        f"model={MODEL if IS_ACP else '(n/a)'} budget={BUDGET} min_gain={MIN_GAIN_PCT}% "
        f"advisor_skill={ADVISOR_SKILL or '(none)'} "
        f"optimizer_skill={OPTIMIZER_SKILL or '(none)'}",
        file=sys.stderr,
    )
    result = run(
        optimize_kernel,
        acp_connector=CONNECTOR if IS_ACP else None,
        out_dir=str(WF_OUT),
        **agent_run_options(),
    )
    # "Out" is a .md path, not the text: see the advisor's port declaration.
    for token in result["Out"] or []:
        print(agent_io.read_text(token))
    # The search always terminates with a best-so-far, so the exit code reports
    # whether the run produced advice at all, not whether it found a speedup --
    # "no gain" is a legitimate, informative outcome.
    return 0 if result["Out"] else 1


if __name__ == "__main__":
    sys.exit(main())
