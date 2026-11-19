#!/usr/bin/env python3
# coding: utf-8
"""Node: `ArtifactGate` -- check the plan declares what the skill promised."""

from __future__ import annotations

from pathlib import Path
import sys

from src.lib.optimize_config import REQUIRED_ARTIFACTS
from wfpy import Port, action, task


@task
class ArtifactGate:
    """The orchestrator's one real job, as a node.

    cannbot's tuning flow is driven by an orchestrator whose own skill says it
    "不负责具体优化建议或代码修改" -- it does not advise or edit. What it does is
    load a phase skill, **verify the artifacts that skill declared**, and only
    then allow the iteration to proceed: "⛔ 核查规则：子技能声明的所有分析制品均已
    生成后，才允许进入 ITER 循环". In a dataflow graph that check is a node, and
    the rest of the orchestrator is the topology.

    What it verifies, in `tune-frontend`'s own order: a Stage A global analysis,
    a Stage B local analysis, a ranked item list, and a `CHOSEN: F-<n>` line.

    An incomplete plan is annotated rather than sent back. A re-ask would need
    the planner to accept a token on either of two input ports, and a wfpy
    `@agent` fires only when EVERY input port holds a token
    (`_agent_staging_runtime._dequeue_agent_inputs`) -- there is no OR-fan-in to
    build a retry edge from. So the directive instead carries an explicit
    INCOMPLETE verdict and the applier's prompt tells it to change nothing in
    that case: no edit is applied, which is what the gate is there to prevent,
    the round is still accounted for, and the reason reaches the report.
    """

    _n: int = 0

    class Ports:
        Plan = Port[str](direction="in")
        Source = Port[str](direction="in")
        Directive = Port[str](direction="out")

    @staticmethod
    def _missing(plan: str) -> list[str]:
        low = plan.lower()
        missing = [
            label for label, needles in REQUIRED_ARTIFACTS
            if not any(n in low for n in needles)
        ]
        if "chosen:" not in low:
            missing.append("a `CHOSEN: F-<n>` line")
        return missing

    # Two inputs consumed together, which for a @task is ordinary AND-firing:
    # one plan and one source arrive per round, and the directive needs both.
    @action(consumes={"Plan": 1, "Source": 1}, produces={"Directive": 1})
    def check(self, plan: str, source: str) -> str:
        self._n += 1
        missing = self._missing(plan)
        verdict = (
            "PLAN INCOMPLETE -- missing: " + "; ".join(missing)
            if missing
            else "PLAN COMPLETE"
        )
        print(f"[opt] gate {self._n}: {verdict}", file=sys.stderr)

        header = [f"Directive {self._n}: {verdict}", ""]
        if missing:
            header += [
                "The analysis below did not declare every artifact the tuning",
                "skill requires, so no optimization may be applied from it.",
                "Reply with the source unchanged.",
                "",
            ]
        try:
            text = Path(str(source)).read_text(encoding="utf-8")
        except OSError:
            text = "(source unavailable)"
        return (
            "\n".join(header)
            + plan
            + f"\n\n--- file to edit: {source} ---\n{text}"
        )
