#!/usr/bin/env python3
# coding: utf-8
"""Phase 4 -- a bounded repair loop around a failing PyPTO kernel.

A loop in wfpy is an ordinary feedback edge. There is no backpressure, so the
loop *must* terminate by itself: here a budget node stops it after N attempts.

    "Kernel" --> Feed --> Prepare --> RunSim --> Gate --(Ok)------> Report
                  ^                                |
                  |                              (Bad)
                  |                                v
                  |                             Budget --(Exhausted)--> Report
                  |                                |
                  |                            (Repair)
                  |                                v
                  +-- ApplyPatch <-- RepairAgent --+

Why each node exists:

- **Feed** merges the initial kernel with retries. An input port does not fan in,
  so "Kernel" and the retry edge cannot both land on Prepare.Kernel -- Feed takes
  two ports and one action each.
- **Budget** holds the attempt counter and is the loop's only exit. The guarded
  action repairs while attempts remain; the unguarded fallback gives up.
- **RepairAgent** is an @agent, so the LLM is just another actor.
- **ApplyPatch** writes the agent's reply to a new file, keeping the original
  kernel untouched.

Transport defaults to ``mock`` so the graph is testable offline, which also
proves the loop terminates. Set ``PYPTO_REPAIR_TRANSPORT=acp`` for a real
repair via the ``claude`` connector (that one spends tokens):

    PYPTO_REPAIR_TRANSPORT=acp ./run.sh repair runs/broken/01_wrong_math.py

Run:
    ./run.sh repair <kernel.py>        # mock: shows the loop hitting its budget
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nodes import WF_OUT, Gate, Prepare, RunResult, RunSim, default_run_mode  # noqa: E402
from wfpy import Port, action, agent, connect, guard, run, task, workflow  # noqa: E402

BUDGET = int(os.environ.get("PYPTO_REPAIR_BUDGET", "2"))
TRANSPORT = os.environ.get("PYPTO_REPAIR_TRANSPORT", "mock")
# Which ACP connector the agent spawns on the acp transport; `wfpy connectors`
# lists what is on the PATH. Ignored by the mock transport.
CONNECTOR = os.environ.get("PYPTO_REPAIR_CONNECTOR", "claude")
IS_ACP = TRANSPORT == "acp"
# A full-file rewrite takes longer than wfpy's 2-minute default.
TIMEOUT_MS = int(os.environ.get("PYPTO_REPAIR_TIMEOUT_MS", "600000"))

# The cannbot pypto-* skills all target Tensor mode, so they apply to these
# kernels directly. wfpy resolves `skill=<name>` against WF_SKILLS_ROOT (its
# highest-priority root), which lets the skills be used in place: a symlink
# would be rejected, because _resolve_skill_dir() resolves the path and then
# requires it to still sit under the root.
SKILLS_ROOT = Path(os.environ.get("WF_SKILLS_ROOT", "/home/endrix/git/pypto/cannbot-skills/ops"))
SKILL_NAME = os.environ.get("PYPTO_REPAIR_SKILL", "pypto-precision-debug")
# Only claim the skill if it is really there; wfpy raises if it cannot resolve.
REPAIR_SKILL = SKILL_NAME if (SKILLS_ROOT / SKILL_NAME / "SKILL.md").is_file() else None
if REPAIR_SKILL:
    os.environ.setdefault("WF_SKILLS_ROOT", str(SKILLS_ROOT))

REPAIR_PROMPT = """You are fixing a PyPTO (Ascend NPU) Tensor-mode kernel that fails
its own numeric self-check against a pure-torch reference.

Rules:
- Reply with the COMPLETE corrected Python file and nothing else. No prose, no fences.
- Change only what is needed to make the numeric check pass.
- Keep the `--run_mode {sim,npu}` flag, the PASS:/FAIL: prints and the exit codes.
- `pypto.mean` does not exist in this version; use `pypto.sum`.
- Set vector tiles with `pypto.set_vec_tile_shapes` before vector ops and cube
  tiles with `pypto.set_cube_tile_shapes` before `pypto.matmul`.
- For a reduction, the tile must span the whole reduced axis.
"""


@dataclass(frozen=True)
class Attempt:
    """A failure handed to the repair agent."""

    kernel: str
    name: str
    attempt: int
    error: str
    source: str


@task
class Feed:
    """Merge the initial kernel with retried ones. Two ports: inputs do not fan in."""

    class Ports:
        Initial = Port[str](direction="in")
        Retry = Port[str](direction="in")
        Out = Port[str](direction="out")

    @action(consumes={"Initial": 1}, produces={"Out": 1})
    def from_initial(self, kernel: str) -> str:
        return kernel

    @action(consumes={"Retry": 1}, produces={"Out": 1})
    def from_retry(self, kernel: str) -> str:
        return kernel


@task
class Budget:
    """The loop's only exit: repair while attempts remain, else give up."""

    budget: int

    _attempts: int = 0

    class Ports:
        Bad = Port[RunResult](direction="in")
        Repair = Port[str](direction="out")
        Exhausted = Port[RunResult](direction="out")

    @action(consumes={"Bad": 1}, produces={"Repair": 1})
    @guard(lambda self, result: self._attempts < self.budget)
    def ask_repair(self, result: RunResult) -> str:
        self._attempts += 1
        try:
            source = Path(result.job.kernel).read_text(encoding="utf-8")
        except OSError:
            source = "(source unavailable)"
        print(
            f"[repair] attempt {self._attempts}/{self.budget} on {result.job.name}: "
            f"{result.error[:60]}",
            file=sys.stderr,
        )
        # The agent port is a plain string, so the attempt is rendered into one.
        return (
            f"Attempt {self._attempts} of {self.budget}.\n"
            f"Kernel: {result.job.kernel}\n"
            f"Failure: {result.error}\n\n"
            f"--- current source ---\n{source}"
        )

    @action(consumes={"Bad": 1}, produces={"Exhausted": 1})
    def give_up(self, result: RunResult) -> RunResult:  # no guard -- the fallback
        return result


# NOTE: `wfpy plan` REWRITES this decorator in place, flattening these constants
# into literals and appending its own camelCase defaults. That silently undoes
# the env-var wiring below -- a previous commit shipped a hardcoded
# transport="mock" while the startup banner still printed TRANSPORT, so
# PYPTO_REPAIR_TRANSPORT=acp claimed to use the agent and did not. If you run
# `./run.sh plan` on this file, re-check the decorator afterwards.
@agent(
    prompt=REPAIR_PROMPT,
    transport=TRANSPORT,
    connector=CONNECTOR if IS_ACP else "",
    timeout_ms=TIMEOUT_MS,
    skill=REPAIR_SKILL,  # None when the cannbot skills are not on disk
    # The mock reply is deliberately NOT a fix: the loop then runs to its budget,
    # which is what makes termination observable offline.
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
        # Strip a markdown fence if the agent added one despite the prompt.
        text = patched.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1 : -1 if lines[-1].strip() == "```" else len(lines)])
        target.write_text(text + "\n", encoding="utf-8")
        return str(target)


@task
class Report:
    """Terminal node. Ok and Exhausted are separate ports."""

    class Ports:
        Ok = Port[RunResult](direction="in")
        Exhausted = Port[RunResult](direction="in")
        Out = Port[str](direction="out")

    @action(consumes={"Ok": 1}, produces={"Out": 1})
    def on_ok(self, result: RunResult) -> str:
        return f"REPAIRED  {result.job.name}: now passes ({result.seconds}s, {result.job.kernel})"

    @action(consumes={"Exhausted": 1}, produces={"Out": 1})
    def on_exhausted(self, result: RunResult) -> str:
        # The name is the *last* attempt's file, not the original kernel: each
        # retry runs a patched copy, so say so rather than implying otherwise.
        return (
            f"GAVE UP   after {BUDGET} attempt(s); last tried {result.job.name} "
            f"-- {result.error[:60]}"
        )


@workflow(inputs={"Kernel": str}, outputs={"Out": str})
def repair_kernel() -> None:
    """Wire the repair loop, including the feedback edge."""
    feed = Feed()
    prepare = Prepare(run_mode=default_run_mode())
    runner = RunSim()
    gate = Gate()
    budget = Budget(budget=BUDGET)
    repair = RepairAgent()
    patch = ApplyPatch()
    report = Report()

    connect("Kernel", feed.Initial)
    connect(feed.Out, prepare.Kernel)
    connect(prepare.Job, runner.Job)
    connect(runner.Result, gate.Result)

    connect(gate.Ok, report.Ok)
    connect(gate.Bad, budget.Bad)
    connect(budget.Exhausted, report.Exhausted)

    # The feedback edge: repair -> patch -> back into Feed.
    connect(budget.Repair, repair.Failure)
    connect(repair.Patched, patch.Patched)
    connect(patch.Kernel, feed.Retry)

    connect(report.Out, "Out")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: phase4_repair.py <kernel.py>", file=sys.stderr)
        return 2
    print(
        f"[repair] transport={TRANSPORT} budget={BUDGET} skill={REPAIR_SKILL or '(none)'}",
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
