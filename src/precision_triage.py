#!/usr/bin/env python3
# coding: utf-8
"""Find which op in a PyPTO kernel first computes the wrong value.

Four nodes, two of them sources and one a sub-workflow three levels deep:

    KernelSource(suspect)   --+
                              |--> triage_flow --+--> MarkdownViewer
    KernelSource(reference) --+                  |
                                                 +--> "Out"

Differential triage. Both kernels are run with PyPTO's pass verification on,
which dumps the output tensor of **every** op; the two runs' dumps are aligned
by op magic and compared in execution order, and the earliest op whose output
differs is the fault. Everything after it is a consequence.

Why a reference kernel is required: a wrong reduction axis does **not** change
the op sequence -- good and broken builds of the attention kernel both dumped 20
identically-named ops -- so there is nothing structural to find. The fault is in
the values, and values need something to be wrong against.

Why it runs on the NPU: pass verification copies a tensor off the device, and in
sim the run dies in `setup_verify_data` with RT_MEMCPY_FAILED. Not a preference.

What it found on the shipped fixture, whose softmax reduces dim=0 instead of
dim=-1: ops A_MUL_B and MULS agree to 0.000000, and the first divergence is
ROWMAX_SINGLE -- the softmax's row maximum, which is exactly where the axis is
read. The matmul and the scaling were never in question.

Run:
    ./run.sh triage                                     # the shipped fixture
    ./run.sh triage <suspect.py> <reference.py>
"""

import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib import agent_io, cann  # noqa: E402
from src.lib.run_options import agent_run_options, summary as kept_summary  # noqa: E402
from src.lib.runner import REPO_ROOT, WF_OUT  # noqa: E402
from src.lib.triage_config import (  # noqa: E402
    ANALYST_SKILL,
    CONNECTOR,
    IS_ACP,
    MODEL,
    RUN_MODE,
    TRANSPORT,
)
from src.sources.kernel_source import KernelSource  # noqa: E402
from src.viewers.markdown_viewer import MarkdownViewer  # noqa: E402
from src.workflows.triage_flow import triage_flow  # noqa: E402
from wfpy import File, connect, run, workflow  # noqa: E402

DEFAULT_SUSPECT = "fixtures/05_attention_block_broken.py"
DEFAULT_REFERENCE = "examples/05_attention_block.py"


def _requested(env: str, argv_index: int, default: str) -> str:
    """A kernel path from the environment, else argv, else the default.

    Not argv alone: a source's path is a constructor parameter, so the graph
    needs it at BUILD time, and whoever imports this module owns argv -- the
    wfpy IDE extension passes its own flags, which would be read as paths.
    """
    from_env = os.environ.get(env, "").strip()
    if from_env:
        return from_env
    if (
        __name__ == "__main__"
        and len(sys.argv) > argv_index
        and not sys.argv[argv_index].startswith("-")
    ):
        return sys.argv[argv_index]
    return default


def _abs(path: str) -> str:
    return path if Path(path).is_absolute() else str(REPO_ROOT / path)


SUSPECT = _abs(_requested("PYPTO_SUSPECT", 1, DEFAULT_SUSPECT))
REFERENCE = _abs(_requested("PYPTO_REFERENCE", 2, DEFAULT_REFERENCE))


@workflow(inputs={}, outputs={"Out": File(ext=".md"), "Verdict": str})
def precision_triage() -> None:
    """Four nodes: the two kernels, the triage, and somewhere to read it."""
    suspect = KernelSource(path=File(SUSPECT))
    reference = KernelSource(path=File(REFERENCE))
    triage = triage_flow()
    table = MarkdownViewer()
    analysis = MarkdownViewer()

    connect(suspect.Out, triage.Suspect)
    connect(reference.Out, triage.Reference)

    # Two viewers, one per document: the op table is evidence, the analysis is
    # the reading of it, and an output port broadcasts, so "Out" gets the
    # analysis as well for the terminal.
    connect(triage.Report, table.In)
    connect(triage.Analysis, analysis.In)
    connect(triage.Analysis, "Out")
    connect(triage.Verdict, "Verdict")


def main() -> int:
    # Makes an IDE run behave like a run.sh run: run.sh sources set_env.sh, a
    # direct launch from an IDE does not.
    cann.apply_to_process()
    if IS_ACP:
        # wfpy reports the model a session accepted at INFO ("Session model set
        # to ..."), and a model the connector does not offer is a warning it
        # otherwise swallows. Without this the run looks the same either way.
        logging.basicConfig(level=logging.INFO, format="[wfpy:%(levelname)s] %(message)s")
        logging.getLogger("wfpy").setLevel(logging.INFO)
    print(
        f"[triage] suspect={SUSPECT}\n"
        f"[triage] reference={REFERENCE}\n"
        f"[triage] keeping={kept_summary()}\n"
        f"[triage] mode={RUN_MODE} (pinned) transport={TRANSPORT} "
        f"model={MODEL if IS_ACP else '(n/a)'} skill={ANALYST_SKILL or '(none)'}",
        file=sys.stderr,
    )
    result = run(
        precision_triage,
        acp_connector=CONNECTOR if IS_ACP else None,
        out_dir=str(WF_OUT),
        **agent_run_options(),
    )
    # "Out" is a .md path now, not the text: the analyst's port declares a
    # format, so wfpy wrote the reply to a file and passed that on.
    for token in result["Out"] or []:
        print(agent_io.read_text(token))
    verdicts = result["Verdict"] or []
    return 0 if verdicts and all(v.startswith(("CULPRIT", "NO DIVERGENCE")) for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
