"""Resume a PANKO intent tree and run its delivered KDA kernel on the NPU."""

import os
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib import cann
from src.lib.runner import DEVICES, REPO_ROOT, WF_OUT
from src.lib.run_options import agent_run_options
from src.lib.skill_paths import ops_root
from src.sources.kernel_source import KernelSource
from src.tasks.panko_dispatch import PankoDispatch
from src.viewers.markdown_viewer import MarkdownViewer
from wfpy import File, connect, run, workflow

os.environ.setdefault("WF_SKILLS_ROOT", str(ops_root()))

from src.agents.panko_coder import PankoCoder
from src.agents.panko_tuner import PankoTuner


@workflow(inputs={}, outputs={"Out": File(ext=".md")})
def panko_search() -> None:
    workspace = os.environ.get("PYPTO_PANKO_WORKSPACE") or REPO_ROOT / "custom/panko" / uuid4().hex
    seed = KernelSource(path=File(str(REPO_ROOT / "examples/kda_decay_matrix_verified.py")))
    dispatch = PankoDispatch(workspace=str(Path(workspace).resolve()), device=DEVICES[0],
                             budget=int(os.environ.get("PYPTO_PANKO_BUDGET", "6")),
                             tile_search=os.environ.get("PYPTO_PANKO_TILE_SEARCH", "1") != "0")
    tuner = PankoTuner()
    coder = PankoCoder()
    viewer = MarkdownViewer()
    frontier = MarkdownViewer()
    connect(seed.Out, dispatch.Seed)
    connect(dispatch.Tune, tuner.Request)
    connect(tuner.Directive, dispatch.Directive)
    connect(dispatch.Code, coder.Request)
    connect(coder.Source, dispatch.Source)
    connect(dispatch.Frontier, frontier.In)
    connect(dispatch.Report, viewer.In)
    connect(dispatch.Report, "Out")


def main() -> int:
    cann.apply_to_process()
    reports = run(panko_search, out_dir=str(WF_OUT), **agent_run_options())["Out"] or []
    for report in reports:
        print(Path(str(report)).read_text())
    return 0 if reports and all("**VERIFIED**" in Path(str(r)).read_text().splitlines()[:3]
                                for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
