"""Generate competing residual RMSNorm kernels and compare measured device latency."""

import os
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib import cann
from src.lib.combo_config import CONNECTOR, IS_ACP
from src.lib.residual_contract import REFERENCE
from src.lib.run_options import agent_run_options
from src.lib.runner import DEVICES, REPO_ROOT, WF_OUT
from src.sources.kernel_source import KernelSource
from src.tasks.compare_residual import CompareResidual
from src.viewers.markdown_viewer import MarkdownViewer
from src.workflows.residual_branch import residual_author, residual_branch
from wfpy import File, connect, run, workflow


@workflow(inputs={}, outputs={"Out": File(ext=".md")})
def residual_rmsnorm():
    workspace = str(REPO_ROOT / "custom" / "residual_rmsnorm" / uuid4().hex)
    budget = int(os.environ.get("PYPTO_DEVICE_TUNE_BUDGET", "3"))
    if budget < 0:
        raise ValueError("PYPTO_DEVICE_TUNE_BUDGET must be nonnegative")
    device = DEVICES[0] if DEVICES else 0
    source = KernelSource(path=File(str(REFERENCE)))
    planner = residual_author(
        "plan", f"Read the input unfused residual RMSNorm source and identify structurally similar "
        f"PyPTO Tensor mode examples under {REPO_ROOT / 'examples'}. Record the signature, both outputs, "
        "reduction and broadcast semantics, supported JIT operations and relevant files. Give both "
        "designers the same constraints. The scheduler owns the workspace and writes the plan.",
        "pypto-op-plan",
    )()
    row_reuse = residual_branch("row_reuse", workspace, budget, device)()
    split_reduction = residual_branch("split_reduction", workspace, budget, device)()
    comparison = CompareResidual(workspace=workspace, device=device)
    plan_view = MarkdownViewer()
    verdict_view = MarkdownViewer()

    connect(source.Out, planner.Brief)
    connect(planner.Artifact, plan_view.In)
    connect(planner.Artifact, row_reuse.Plan)
    connect(planner.Artifact, split_reduction.Plan)
    connect(row_reuse.Result, comparison.Row)
    connect(split_reduction.Result, comparison.Split)
    connect(comparison.Report, verdict_view.In)
    connect(comparison.Report, "Out")


def main():
    cann.apply_to_process()
    result = run(residual_rmsnorm, acp_connector=CONNECTOR if IS_ACP else None,
                 out_dir=str(WF_OUT), **agent_run_options())
    reports = result["Out"] or []
    succeeded = bool(reports)
    for path in reports:
        report = Path(str(path)).read_text()
        print(report)
        succeeded = succeeded and report.startswith("# Residual RMSNorm comparison\n\n**COMPLETE**\n")
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
