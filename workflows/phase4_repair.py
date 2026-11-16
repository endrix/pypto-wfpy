#!/usr/bin/env python3
# coding: utf-8
"""Phase 4 -- a bounded repair loop around a failing PyPTO kernel.

Five actors. A loop in wfpy is an ordinary feedback edge with no backpressure, so
it must terminate by itself: ``Decide`` is the only exit.

    "Kernel" --> RunKernel --> Decide --(Done)-------> Report --> "Out"
                   ^  ^                |--(Exhausted)--^
                   |  |                |
                   |  |             (Repair)
                   |  |                v
                   |  +-- ApplyPatch <-- RepairAgent
                   |
                 Initial / Retry: two input ports on one actor, because a wfpy
                 input port does not fan in.

The gate checks more than stdout. A kernel that compiles nothing still exits 0,
so ``rc == 0 and "PASS:" in stdout`` cannot tell a correct kernel from one that
never ran. Each attempt therefore also compiles with per-pass IR dumping and the
gate requires that a kernel was really produced and that the IR reached all four
levels (TENSOR -> TILE -> BLOCK -> EXECUTE).

Structural change is *reported, not failed*. Fixing a tiling bug necessarily
changes the tile shapes -- on the LayerNorm fixture the broken kernel splits into
5 BLOCK_GRAPH leaves and the repaired one into 1 -- so treating any structural
difference as a failure would reject exactly the repair that was asked for.

Run:
    ./run.sh repair <kernel.py>                        # mock: hits the budget
    PYPTO_REPAIR_TRANSPORT=acp ./run.sh repair <kernel.py>   # real; spends tokens
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _graph  # noqa: E402
from _nodes import WF_OUT, RunResult, default_run_mode, execute_job, make_job  # noqa: E402
from wfpy import Port, action, agent, connect, guard, run, task, workflow  # noqa: E402

BUDGET = int(os.environ.get("PYPTO_REPAIR_BUDGET", "2"))
TRANSPORT = os.environ.get("PYPTO_REPAIR_TRANSPORT", "mock")
CONNECTOR = os.environ.get("PYPTO_REPAIR_CONNECTOR", "claude")
IS_ACP = TRANSPORT == "acp"
TIMEOUT_MS = int(os.environ.get("PYPTO_REPAIR_TIMEOUT_MS", "600000"))

# The cannbot pypto-* skills all target Tensor mode. wfpy resolves skill=<name>
# against WF_SKILLS_ROOT, its highest-priority root, which lets them be used in
# place; a symlink would be rejected by its containment check.
SKILLS_ROOT = Path(os.environ.get("WF_SKILLS_ROOT", "/home/endrix/git/pypto/cannbot-skills/ops"))
SKILL_NAME = os.environ.get("PYPTO_REPAIR_SKILL", "pypto-precision-debug")
REPAIR_SKILL = SKILL_NAME if (SKILLS_ROOT / SKILL_NAME / "SKILL.md").is_file() else None
if REPAIR_SKILL:
    os.environ.setdefault("WF_SKILLS_ROOT", str(SKILLS_ROOT))

REPAIR_PROMPT = """You are fixing a PyPTO (Ascend NPU) Tensor-mode kernel that fails
its own numeric self-check against a pure-torch reference.

Rules:
- Reply with the COMPLETE corrected Python file and nothing else. No prose, no fences.
- Change only what is needed to make the check pass. Keep the `--run_mode {sim,npu}`
  flag, the PASS:/FAIL: prints and the exit codes.
- Do not weaken the test: leave the tolerances, the torch reference and every
  assertion exactly as they are.
- `pypto.mean` does not exist in this version; use `pypto.sum` over the axis.
- Set vector tiles with `pypto.set_vec_tile_shapes` before vector ops and cube
  tiles with `pypto.set_cube_tile_shapes` before `pypto.matmul`.
- For a reduction, the tile must span the whole reduced axis.
"""


@dataclass(frozen=True)
class Outcome:
    """One attempt: what it computed, and what the compiler made of it."""

    result: RunResult
    graph: _graph.GraphSummary
    attempt: int

    @property
    def ok(self) -> bool:
        """Numerically right AND actually compiled through the full pipeline."""
        return self.result.passed and not self.graph.problems()

    def why(self) -> str:
        reasons = list(self.graph.problems())
        if not self.result.passed:
            reasons.insert(0, self.result.error or "numeric check failed")
        return "; ".join(reasons) or "ok"


@task
class RunKernel:
    """Resolve, compile with IR dumping, run, and summarize the graph.

    Prepare and RunSim are one actor here: splitting them added a node without
    adding a decision, and pool_validate already runs them merged.
    """

    run_mode: str

    timeout_s: int = 900
    _n: int = 0

    class Ports:
        Initial = Port[str](direction="in")
        Retry = Port[str](direction="in")
        Result = Port[Outcome](direction="out")

    def _run(self, kernel: str) -> Outcome:
        self._n += 1
        job = make_job(kernel, self.run_mode, suffix=f"_repair{self._n}",
                       slot=self._n - 1, via_ir_runner=True)
        # Only the four passes that expose the IR level transitions, not all 39.
        job = type(job)(**{**job.__dict__, "cmd": job.cmd + ("--passes", _graph.DUMP_PASSES)})
        result = execute_job(job, self.timeout_s)
        summary = _graph.summarize(job.work_dir)
        outcome = Outcome(result=result, graph=summary, attempt=self._n)
        # "run", not "attempt": this counts kernel runs, while Decide counts
        # repair requests. A budget of N means N repairs and so N+1 runs.
        print(
            f"[repair] run {self._n}: {Path(kernel).stem} -> "
            f"{'ok' if outcome.ok else outcome.why()} "
            f"[compiled={summary.compiled} leaves={summary.block_leaves}]",
            file=sys.stderr,
        )
        return outcome

    @action(consumes={"Initial": 1}, produces={"Result": 1})
    def from_initial(self, kernel: str) -> Outcome:
        return self._run(kernel)

    @action(consumes={"Retry": 1}, produces={"Result": 1})
    def from_retry(self, kernel: str) -> Outcome:
        return self._run(kernel)


@task
class Decide:
    """Accept, ask for a repair, or give up. The loop's only exit.

    The three guards are mutually exclusive on purpose. wfpy picks
    nondeterministically among actions that are all activated -- declaration
    order happens to break the tie in this runtime, but the model does not
    promise it -- so each action states its full condition rather than relying on
    being declared after a guarded one.
    """

    budget: int

    _attempts: int = 0
    _baseline: Any = None  # GraphSummary of the first run, for the structural delta

    class Ports:
        Result = Port[Outcome](direction="in")
        Repair = Port[str](direction="out")
        Done = Port[Outcome](direction="out")
        Exhausted = Port[Outcome](direction="out")

    def _remember(self, outcome: Outcome) -> None:
        if self._baseline is None:
            self._baseline = outcome.graph

    @action(consumes={"Result": 1}, produces={"Done": 1})
    @guard(lambda self, outcome: outcome.ok)
    def accept(self, outcome: Outcome) -> Outcome:
        self._remember(outcome)
        return outcome

    @action(consumes={"Result": 1}, produces={"Repair": 1})
    @guard(lambda self, outcome: not outcome.ok and self._attempts < self.budget)
    def ask_repair(self, outcome: Outcome) -> str:
        self._remember(outcome)
        self._attempts += 1
        try:
            source = Path(outcome.result.job.kernel).read_text(encoding="utf-8")
        except OSError:
            source = "(source unavailable)"
        return (
            f"Attempt {self._attempts} of {self.budget}.\n"
            f"Kernel: {outcome.result.job.kernel}\n"
            f"Failure: {outcome.why()}\n\n"
            f"--- current source ---\n{source}"
        )

    @action(consumes={"Result": 1}, produces={"Exhausted": 1})
    @guard(lambda self, outcome: not outcome.ok and self._attempts >= self.budget)
    def give_up(self, outcome: Outcome) -> Outcome:
        self._remember(outcome)
        return outcome

    def delta(self, outcome: Outcome) -> list[str]:
        if self._baseline is None or self._baseline is outcome.graph:
            return []
        return outcome.graph.differences(self._baseline)


@agent(
    prompt=REPAIR_PROMPT,
    transport=TRANSPORT,
    connector=CONNECTOR if IS_ACP else "",
    timeout_ms=TIMEOUT_MS,
    skill=REPAIR_SKILL,  # None when the cannbot skills are not on disk
    # The mock reply is deliberately NOT a fix, so the loop runs to its budget
    # and termination stays observable offline.
    mock_outputs={"Patched": "# mock transport: no real repair was attempted\n"},
)
class RepairAgent:
    class Ports:
        Failure = Port[str](direction="in")
        Patched = Port[str](direction="out")


@task
class ApplyPatch:
    """Write the agent's reply beside the original, never over it."""

    _n: int = 0

    class Ports:
        Patched = Port[str](direction="in")
        Kernel = Port[str](direction="out")

    @action(consumes={"Patched": 1}, produces={"Kernel": 1})
    def apply(self, patched: str) -> str:
        self._n += 1
        out_dir = Path(__file__).resolve().parent.parent / "runs" / "repaired"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"attempt{self._n}.py"
        text = patched.strip()
        if text.startswith("```"):  # strip a fence the agent added despite the prompt
            lines = text.splitlines()
            text = "\n".join(lines[1 : -1 if lines[-1].strip() == "```" else len(lines)])
        target.write_text(text + "\n", encoding="utf-8")
        return str(target)


@task
class Report:
    """Terminal node. Done and Exhausted are separate ports: inputs do not fan in."""

    class Ports:
        Done = Port[Outcome](direction="in")
        Exhausted = Port[Outcome](direction="in")
        Out = Port[str](direction="out")

    @staticmethod
    def _graph_line(outcome: Outcome) -> str:
        g = outcome.graph
        return (f"      graph: compiled={g.compiled} levels={len(g.levels)}/4 "
                f"leaves={g.block_leaves} tiles={len(g.tile_shapes)}")

    @action(consumes={"Done": 1}, produces={"Out": 1})
    def on_done(self, outcome: Outcome) -> str:
        head = (f"REPAIRED  {outcome.result.job.name}: passes and compiled clean "
                f"({outcome.result.seconds}s, attempt {outcome.attempt})")
        return "\n".join([head, self._graph_line(outcome)])

    @action(consumes={"Exhausted": 1}, produces={"Out": 1})
    def on_exhausted(self, outcome: Outcome) -> str:
        head = (f"GAVE UP   after {BUDGET} repair attempt(s); last tried "
                f"{outcome.result.job.name}\n      reason: {outcome.why()}")
        return "\n".join([head, self._graph_line(outcome)])


@workflow(inputs={"Kernel": str}, outputs={"Out": str})
def repair_kernel() -> None:
    """Five actors, with the feedback edge closing back onto RunKernel.Retry."""
    runner = RunKernel(run_mode=default_run_mode())
    decide = Decide(budget=BUDGET)
    repair = RepairAgent()
    patch = ApplyPatch()
    report = Report()

    connect("Kernel", runner.Initial)
    connect(runner.Result, decide.Result)

    connect(decide.Done, report.Done)
    connect(decide.Exhausted, report.Exhausted)

    # The loop: repair -> patch -> back into the runner's second input port.
    connect(decide.Repair, repair.Failure)
    connect(repair.Patched, patch.Patched)
    connect(patch.Kernel, runner.Retry)

    connect(report.Out, "Out")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: phase4_repair.py <kernel.py>", file=sys.stderr)
        return 2
    print(
        f"[repair] transport={TRANSPORT} budget={BUDGET} mode={default_run_mode()} "
        f"skill={REPAIR_SKILL or '(none)'}",
        file=sys.stderr,
    )
    result = run(
        repair_kernel,
        inputs={"Kernel": sys.argv[1]},
        acp_connector=CONNECTOR if IS_ACP else None,
        out_dir=str(WF_OUT),
    )
    lines = result["Out"] or ["(no output)"]
    for line in lines:
        print(line)
    return 0 if any(ln.startswith("REPAIRED") for ln in lines) else 1


if __name__ == "__main__":
    sys.exit(main())
