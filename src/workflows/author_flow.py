#!/usr/bin/env python3
# coding: utf-8
"""Sub-workflow factory: author a PyPTO kernel from PyTorch, stage by stage.

    "Source" -> intent -> gate -> api -> gate -> golden -> gate
                                -> design -> gate -> develop -> gate -> report

Every token between nodes is a FILE. The source is the PyTorch module, each
stage writes one artifact, and each gate writes the brief the next stage reads.
Nothing is a bare string: wfpy stages a file input as a real file for an ACP
agent rather than inlining it into the prompt, so a stage opens a document.

cannbot's kernel-development stage machine. Five stages, each loading one skill
and leaving one declared artifact in a shared workspace; a gate after each one
checks the artifact exists and is more than a stub before the next stage runs.

**The graph is the orchestrator.** The skill that normally drives this says of
itself that it does not advise or edit -- it loads a phase skill, verifies that
phase's artifacts, and advances a state machine. Topology does the advancing and
`stage_gate` does the verifying, so there is no orchestrator agent and no stage
decides what comes next.

The stages talk through the WORKSPACE, not through the tokens. Each gate places
its artifact at `custom/<op>/<name>` and each stage's brief names that directory,
because the cannbot skills read `custom/<op>/SPEC.md` from disk rather than being
handed its text. The tokens stay small and the workspace is the shared state.

Costly: five agents, one of them writing a whole kernel, at up to half an hour
each. Nothing stands in for them -- the structure is checked by `wfpy plan` and
the linter, and the rest is what a model does.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.stage_author import stage_author  # noqa: E402
from src.lib import author_config as cfg  # noqa: E402
from src.tasks.verify_kda import VerifyKda
from src.tasks.merge_brief import MergeBrief
from src.tasks.stage_gate import stage_gate  # noqa: E402
from src.tasks.write_author_report import WriteAuthorReport  # noqa: E402
from wfpy import File, connect, workflow  # noqa: E402

# What each gate tells the stage after it. The workspace is the handover.
_NEXT = (
    "Stage {verdict}\n\nWorkspace: {workspace}\nOperator: {op}\n"
    "The artifact just written is {artifact}. Read it, and everything else in "
    "the workspace, before you write yours."
)

# stage name, prompt, skill, artifact extension, artifact name
_PLAN = (
    ("intent", cfg.INTENT_PROMPT, cfg.INTENT_SKILL, ".md", "SPEC.md"),
    ("api", cfg.API_PROMPT, cfg.API_SKILL, ".md", "API_REPORT.md"),
    ("golden", cfg.GOLDEN_PROMPT, cfg.GOLDEN_SKILL, ".py", "{op}_golden.py"),
    ("design", cfg.DESIGN_PROMPT, cfg.DESIGN_SKILL, ".md", "DESIGN.md"),
    ("develop", cfg.DEVELOP_PROMPT, cfg.DEVELOP_SKILL, ".py", "{op}_impl.py"),
)


def author_flow(op: str, workspace: str, torch_source: str, *, verify_kda: bool = False, **agent_options: str) -> Callable[[], None]:
    """Build the authoring workflow for one operator."""

    @workflow(
        inputs={"Source": File(ext=".py")},
        outputs={"Report": File(ext=".md"), "Verdict": str},
    )
    def author_stages() -> None:
        """Five stage/gate pairs in order, then a report over the workspace."""
        report = WriteAuthorReport(workspace=workspace, op=op)

        # The PyTorch module is the first stage's brief; after that, each gate's
        # brief is the next stage's. One port name throughout, because both are
        # files.
        brief = "Source"
        for stage, prompt, skill, ext, artifact in _PLAN:
            prompt = prompt.format(workspace=workspace, torch_source=torch_source,
                                   golden_name="kda_golden" if verify_kda else op,
                                   artifact=Path(workspace) / artifact.format(op=op))
            prompt += f"\nOperator: {op}. Original source: {torch_source}."
            if verify_kda:
                prompt += cfg.KDA_CONTRACT
            checked = verify_kda and stage in ("golden", "develop")
            if checked:
                prompt += ("\nPut only compilable Python in the Artifact output. Progress updates, "
                           "test summaries and Markdown fences cannot appear in that file.")
                repair = MergeBrief(op=op, workspace=workspace, torch_source=torch_source,
                                    artifact=artifact.format(op=op))
                connect(brief, repair.Initial)
                brief = repair.Out
            if verify_kda and stage == "golden":
                prompt += ("\nExport kda_golden(a, k, g, diagonal=1) as the independent FP32 Torch "
                           "reference. The workflow validates it on all five fixtures before design. "
                           "On retry correct the submitted golden using the attached failure.")
            if verify_kda and stage == "develop":
                prompt += (
                    "\nExport kda_run(a, k, g, diagonal, run_mode). Configure the PyPTO JIT for the "
                    "specified mode (sim or npu), run the actual kernel once, and return FP32 output "
                    "on the input device. Keep the CLI behind the main guard. The workflow supplies "
                    "fixtures, golden comparison, and a separate NPU verifier. On retry, fix the "
                    "reported failure, preserving the original operator contract and golden."
                )
            author = stage_author(stage, prompt, skill, ext, **agent_options)()
            gate = stage_gate(stage, artifact, _NEXT)(workspace=workspace, op=op)

            connect(brief, author.Brief)
            connect(author.Artifact, gate.Artifact)
            brief = gate.Brief
            if checked:
                verification = VerifyKda(workspace=workspace, op=op, stage=stage)
                connect(brief, verification.Brief)
                connect(verification.Retry, repair.Retry)
                connect(verification.Failed, report.GoldenFailed if stage == "golden" else report.KernelFailed)
                brief = verification.Passed

        # The last brief closes the chain; the report reads the workspace for
        # what each stage actually left.
        connect(brief, report.Brief)
        connect(report.Report, "Report")
        connect(report.Out, "Verdict")

    return author_stages
