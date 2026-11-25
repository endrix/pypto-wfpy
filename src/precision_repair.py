"""Localize and repair the bundled attention fault using measured NPU checkpoints."""

import os
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfpy import File, connect, run, workflow
from src.lib import cann
from src.lib.combo_config import CONNECTOR, IS_ACP
from src.lib.precision_repair import SUSPECT
from src.lib.run_options import agent_run_options
from src.lib.runner import REPO_ROOT, WF_OUT
from src.sources.kernel_source import KernelSource
from src.viewers.markdown_viewer import MarkdownViewer
from src.workflows.precision_repair_flow import precision_repair_flow


@workflow(inputs={}, outputs={"Out": File(ext=".md"), "Verdict": str})
def precision_repair():
    source = KernelSource(path=File(str(SUSPECT)))
    workspace = REPO_ROOT / "custom" / "precision_repair" / uuid4().hex
    budget = int(os.getenv("PYPTO_PRECISION_REPAIR_BUDGET", "2"))
    loop = precision_repair_flow(str(workspace), budget)()
    viewer = MarkdownViewer()
    connect(source.Out, loop.Suspect)
    connect(loop.Report, viewer.In)
    connect(loop.Report, "Out")
    connect(loop.Verdict, "Verdict")


def main():
    cann.apply_to_process()
    result = run(precision_repair, out_dir=str(WF_OUT),
                 acp_connector=CONNECTOR if IS_ACP else None, **agent_run_options())
    for path in result["Out"] or []:
        print(f"Report: {path}")
    verdicts = result["Verdict"] or []
    print(*verdicts)
    return 0 if verdicts and all(v in ("REPAIRED", "ALREADY PASSING") for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
