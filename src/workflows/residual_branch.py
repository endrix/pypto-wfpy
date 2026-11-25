"""Generate, verify and tune one residual RMSNorm design in an isolated workspace."""

from pathlib import Path

from src.agents.stage_author import stage_author
from src.lib import combo_config as cfg
from src.lib.residual_contract import BRANCHES, REFERENCE
from src.tasks.merge_brief import MergeBrief
from src.tasks.residual_handoff import ResidualHandoff
from src.tasks.residual_result import ResidualResult
from src.tasks.residual_verify import ResidualVerify
from src.viewers.markdown_viewer import MarkdownViewer
from src.workflows.device_tune_flow import make_device_tune_flow
from wfpy import File, connect, workflow


def residual_author(stage, prompt, skill, ext=".md"):
    return stage_author(stage, prompt, skill, ext, transport=cfg.TRANSPORT,
                        connector=cfg.CONNECTOR if cfg.IS_ACP else "", model=cfg.MODEL)


def residual_branch(name, workspace, budget, device):
    directory = str(Path(workspace) / name)
    context = (f" Work only on the {name} design. The workflow owns its workspace {directory}. "
               "Return your entire artifact through the Artifact port. Treat reference, "
               "evaluation code and other workspaces as read-only. ")

    @workflow(inputs={"Plan": File(ext=".md")}, outputs={"Result": File(ext=".json")})
    def generate_residual_branch():
        plan = ResidualHandoff(workspace=directory, filename="PLAN.md")
        design = residual_author("design", context + BRANCHES[name] +
                                 " Produce a concrete design with shapes, tile choices and PyPTO APIs.",
                                 "pypto-op-design")()
        design_doc = ResidualHandoff(workspace=directory, filename="DESIGN.md")
        verify = residual_author("scaffold", context +
                                 "Review this design against both outputs of the frozen reference. "
                                 "Read SPEC.md, residual_rmsnorm_golden.py and eval/test_cases.json in the branch workspace. "
                                 "Specify how to exercise cancellation, row-wise reductions, input "
                                 "preservation and changing shapes. Cite each check and identify "
                                 "implementation constraints for the developer.", "pypto-op-verify")()
        verification_doc = ResidualHandoff(workspace=directory, filename="VERIFICATION.md")
        merge = MergeBrief(op="residual_rmsnorm", workspace=directory,
                           torch_source=str(REFERENCE), artifact="kernel.py")
        develop = residual_author("develop", context +
                                  "Implement the design in compilable Python using PyPTO. Export the "
                                  "requested callable. See examples/04_layer_norm.py for the available "
                                  "reduction and examples/kda_decay_matrix_verified.py for sim and NPU "
                                  "calling conventions. Correct the reported defect on a retry. Return "
                                  "the full Python source as Artifact, without prose or Markdown fences.",
                                  "pypto-op-develop", ".py")()
        gate = ResidualVerify(workspace=directory, device=device)
        tuner = make_device_tune_flow(str(Path(directory) / "tune"),
                                      "src.lib.residual_eval", budget=budget, device=device)()
        result = ResidualResult(workspace=directory)
        design_view = MarkdownViewer()
        verification_view = MarkdownViewer()
        tuning_view = MarkdownViewer()

        connect("Plan", plan.Artifact)
        connect(plan.Brief, design.Brief)
        connect(design.Artifact, design_view.In)
        connect(design.Artifact, design_doc.Artifact)
        connect(design_doc.Brief, verify.Brief)
        connect(verify.Artifact, verification_view.In)
        connect(verify.Artifact, verification_doc.Artifact)
        connect(verification_doc.Brief, merge.Initial)
        connect(gate.Retry, merge.Retry)
        connect(merge.Out, develop.Brief)
        connect(develop.Artifact, gate.Artifact)
        connect(gate.Kernel, tuner.Kernel)
        connect(gate.Failed, result.Failed)
        connect(plan.Failed, result.PlanFailed)
        connect(design_doc.Failed, result.DesignFailed)
        connect(verification_doc.Failed, result.ReviewFailed)
        connect(tuner.Winner, result.Kernel)
        connect(tuner.Report, tuning_view.In)
        connect(tuner.Report, result.Report)
        connect(tuner.Status, result.Status)
        connect(result.Result, "Result")

    return generate_residual_branch
