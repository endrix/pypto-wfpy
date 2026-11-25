#!/usr/bin/env python3
# coding: utf-8
"""Author a PyPTO kernel from a PyTorch module, stage by stage.

Three nodes, one of them the stage machine:

    KernelSource --> author_flow --+--> MarkdownViewer
                                   |
                                   +--> "Out"

Five cannbot skills in order, each leaving one declared artifact in
`custom/<op>/`, each gated before the next runs:

    pypto-intent-understand  -> SPEC.md
    pypto-api-explore        -> API_REPORT.md
    pypto-golden-generate    -> <op>_golden.py
    pypto-op-design          -> DESIGN.md
    pypto-op-develop         -> <op>_impl.py

This is the front of the arc the rest of the project continues: what comes out
can be run by `validate`, measured by `profile`, searched by `optimize` and
localised by `triage`. It is the only workflow here that creates a kernel rather
than examining one.

What it does NOT do: claim the result works. The last gate checks that a file was
written, not that it computes anything -- `./run.sh validate <impl>` answers
that, and the report says so.

Costly for real: five agents, one writing a whole kernel.
Nothing stands in for the agents: `wfpy plan` and the linter check the
structure for nothing, and what is left to learn needs a model.

Run:
    ./run.sh author torch_src/rms_norm.py rms_norm   # spends tokens
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib import cann  # noqa: E402
from src.lib.author_config import (  # noqa: E402
    CONNECTOR,
    IS_ACP,
    MODEL,
    TRANSPORT,
)
from src.lib.run_options import agent_run_options, summary as kept_summary  # noqa: E402
from src.lib.runner import REPO_ROOT, WF_OUT  # noqa: E402
from src.sources.kernel_source import KernelSource  # noqa: E402
from src.viewers.markdown_viewer import MarkdownViewer  # noqa: E402
from src.workflows.author_flow import author_flow  # noqa: E402
from wfpy import File, connect, run, workflow  # noqa: E402

# torch_src/, not torch/: a directory called `torch` beside code that does
# `import torch` is only safe by accident. A regular package wins the path
# scan over a namespace portion, so it does not shadow today -- but adding
# an __init__.py to it would break every torch import in the project.
DEFAULT_SOURCE = "torch_src/rms_norm.py"


def _arg(env: str, index: int, default: str) -> str:
    """From the environment, else argv when this file is the program.

    Not argv alone: the workspace and operator name are constructor parameters,
    so the graph needs them at BUILD time, and whoever imports this module owns
    argv -- the IDE extension passes its own flags.
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


@workflow(inputs={}, outputs={"Out": str})
def new_kernel() -> None:
    """Three nodes: the PyTorch module, the five gated stages, and a viewer.

    `KernelSource` is the file source -- the PyTorch module IS the first stage's
    brief, so there is nothing to compose. It replaced a task that read the file
    and emitted a string: a document belongs in a file token, and wfpy stages a
    file input as a real file for the agent rather than inlining it.
    """
    source = KernelSource(path=File(TORCH_SOURCE))
    author = author_flow(OP, WORKSPACE, TORCH_SOURCE)()
    view = MarkdownViewer()

    connect(source.Out, author.Source)
    connect(author.Report, view.In)
    connect(author.Verdict, "Out")


def main() -> int:
    # Makes an IDE run behave like a run.sh run.
    cann.apply_to_process()
    if not Path(TORCH_SOURCE).is_file():
        print(f"[author] no such PyTorch source: {TORCH_SOURCE}", file=sys.stderr)
        return 2
    print(
        f"[author] source={TORCH_SOURCE}\n"
        f"[author] op={OP} workspace={WORKSPACE}\n"
        f"[author] keeping={kept_summary()}\n"
        f"[author] transport={TRANSPORT} model={MODEL if IS_ACP else '(n/a)'}",
        file=sys.stderr,
    )
    result = run(
        new_kernel,
        acp_connector=CONNECTOR if IS_ACP else None,
        out_dir=str(WF_OUT),
        **agent_run_options(),
    )
    lines = result["Out"] or ["(no output)"]
    for line in lines:
        print(line)
    return 0 if any(ln.startswith("AUTHORED") for ln in lines) else 1


if __name__ == "__main__":
    sys.exit(main())
