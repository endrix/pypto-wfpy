#!/usr/bin/env python3
# coding: utf-8
"""Sub-workflow factory: author a kernel, optimise it, redesign if that fails.

    "Source" --> MergeBrief --> author_flow --> RouteAuthored --Kernel--> optimize_flow
                     ^                               |                         |
                     |                            Stalled                  Report
                     |                               v                         v
                     +---- RedesignAdvisor <---- OuterScore <-------------------+
                                                     |
                                                   Done --> "Report"

An agentic loop around two agentic loops. The body is ten agents: five
authoring stages, then the optimiser's planner/applier/doctor and its advisor.

**Three nested loops, each with its own budget**, because each varies a different
thing:

- the optimiser's inner repair loop varies a tiling the compiler REJECTED
  (`repair_budget`),
- its round loop varies the tiling of a fixed design (`budget`),
- and this one varies the DESIGN (`OUTER_BUDGET`). When every tiling of a design
  is rejected or slower, the lever is not another tiling.

That last distinction is the reason this workflow exists rather than just
chaining the two. cannbot's own flow allows it -- Stage 7 can send the work back
-- and nothing here did until now.

`MergeBrief` is at the head because a wfpy input port does not fan in and
`author_flow`'s boundary has one `Source` port, so the first brief and the
redesign cannot share it.

Costly, and linear in `OUTER_BUDGET`: one round is ten agents, one of them
writing a whole kernel. A default run is "author, optimise, and reconsider once
if that found nothing".
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.redesign_advisor import make_redesign_advisor  # noqa: E402
from src.lib.runner import REPO_ROOT
from src.lib.author_config import KDA_CONTRACT
from src.lib.combo_config import OUTER_BUDGET  # noqa: E402
from src.tasks.merge_brief import MergeBrief  # noqa: E402
from src.tasks.outer_score import OuterScore  # noqa: E402
from src.tasks.route_authored import RouteAuthored  # noqa: E402
from src.tasks.write_viewer_text import WriteViewerText
from src.viewers.markdown_viewer import MarkdownViewer
from src.workflows.author_flow import author_flow  # noqa: E402
from src.workflows.optimize_flow import make_optimize_flow  # noqa: E402
from wfpy import File, connect, workflow  # noqa: E402


def combo_flow(op: str, workspace: str, torch_source: str, **agent_options: str) -> Callable[[], None]:
    """Build the author-then-optimise loop for one operator."""

    kda = Path(torch_source).resolve() == REPO_ROOT / "torch_src/kda_decay_matrix.py"

    @workflow(
        inputs={"Source": File(ext=".py")},
        outputs={
            "Report": File(ext=".md"),
            # Through to the top, so the run can be looked at once the agents
            # have finished -- which is the only way to tell what they did.
            "Advice": File(ext=".md"),
            "Swimlane": File(ext=".json"),
            **{lv: File(ext=".json") for lv in ("TensorGraph", "TileGraph",
                                                "BlockGraph", "ExecuteGraph")},
        },
    )
    def flow() -> None:
        """Six nodes, two of them whole workflows, one loop across both."""
        merge = MergeBrief(op=op, workspace=workspace, torch_source=torch_source, artifact="SPEC.md")
        author = author_flow(op, workspace, torch_source, verify_kda=kda, **agent_options)()
        verdict_file = WriteViewerText(directory=str(Path(workspace) / "viewers"), label="Verdict")
        verdict_view = MarkdownViewer()
        route = RouteAuthored(workspace=workspace, op=op)
        search = make_optimize_flow(workspace=workspace, kda_op=op if kda else "", **agent_options)()
        score = OuterScore(budget=OUTER_BUDGET, op=op, workspace=workspace, torch_source=torch_source)
        advisor = make_redesign_advisor(contract=KDA_CONTRACT if kda else "", **agent_options)()

        connect("Source", merge.Initial)
        connect(merge.Out, author.Source)
        connect(author.Report, route.Report)
        connect(author.Verdict, verdict_file.Text)
        connect(verdict_file.Document, verdict_view.In)

        # Authored: optimise it. Not authored: that is evidence too.
        connect(route.Kernel, search.Kernel)
        connect(route.Stalled, score.Stalled)
        connect(search.Report, score.Searched)
        # The advisor's analysis was being produced and dropped: it is evidence
        # for the redesign, and the thing a reader most wants afterwards.
        connect(search.Advice, score.Advice)

        # The feedback edge, and the only thing that can restart authoring.
        connect(score.Evidence, advisor.Evidence)
        connect(advisor.Brief, merge.Retry)

        connect(score.Done, "Report")

        connect(search.Advice, "Advice")
        connect(search.Swimlane, "Swimlane")
        for level in ("TensorGraph", "TileGraph", "BlockGraph", "ExecuteGraph"):
            connect(getattr(search, level), level)

    return flow
