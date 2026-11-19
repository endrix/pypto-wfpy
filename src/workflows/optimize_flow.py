#!/usr/bin/env python3
# coding: utf-8
"""Sub-workflow: a bounded agentic search for a faster kernel.

    "Kernel" -> MergeKernel -> simulate_flow -> analyze_flow -> ScorePerf
                    ^                                             |
                    |                        Request              | Broken      | Result
                    |                           v                 v             v
                    |                      TunePlanner        TileDoctor   WriteOptReport
                    |                           |                 |             |
                    |                      ArtifactGate           |        Report | Analysis
                    |                           |                 |             |      v
                    +-- ApplyCandidate <-- TuneApplier <-----------+          "Report"  PerfAdvisor
                                                                                           |
                                                                                       "Advice"

Eleven nodes: two sub-workflows reused whole, four agents, five tasks. The shape
follows cannbot's tuning orchestrator, with one substitution -- **the graph is
the orchestrator**. `tune-orchestrator`'s own skill says it
"不负责具体优化建议或代码修改": it loads a phase skill, verifies that skill's
declared artifacts, and advances a state machine. Topology and guards do all of
that here, so there is no orchestrator agent. The agents only analyse, decide,
edit and diagnose.

Why each node is separate:

- `MergeKernel` -- a wfpy input port does not fan in, and the loop has to feed a
  sub-workflow boundary with one `Kernel` port.
- `TunePlanner` / `TuneApplier` -- `tune-frontend`'s contract is three ORDERED
  stages, editing forbidden until the analyses exist. One agent asked for a
  finished file fights that; measured, it produced two kernels the compiler
  rejected. Plan and apply as separate nodes make the node contract match the
  skill contract.
- `ArtifactGate` -- the orchestrator's verification step, and the only thing it
  really does.
- `TileDoctor` -- a candidate the compiler REJECTS is a different event from a
  candidate that is merely slower. It gets its own agent, its own skill
  (`pypto-general-debug`) and its own budget.
- `ScorePerf` -- the only exit. One routing function decides between ask,
  diagnose, abort and stop, so the four are exclusive by construction rather
  than by four guards agreeing.
- `PerfAdvisor` -- outside the loop on purpose, so a slow or wrong analysis
  cannot stall the search or steer it.

Termination: a wfpy feedback edge has no backpressure, so every cycle needs a
counter that only goes up. `budget` bounds optimization rounds and
`repair_budget` bounds compiler-rejection repairs; a candidate is adopted only
if it passes its own self-check AND beats the incumbent by more than
`MIN_GAIN_PCT`.

Not yet here, and needed for the full cannbot flow: a performance TARGET and an
early exit when it is met; S1_SETUP as a gate; S2_COLLECT's `debug_options` and
its stdout-to-`runtime_options` contract; the swimlane and in-core phases, which
must re-collect their own data on entry rather than reuse this baseline.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.perf_advisor import PerfAdvisor  # noqa: E402
from src.agents.tile_doctor import TileDoctor  # noqa: E402
from src.agents.tune_applier import TuneApplier  # noqa: E402
from src.agents.tune_planner import TunePlanner  # noqa: E402
from src.lib.optimize_config import BUDGET  # noqa: E402
from src.tasks.apply_candidate import ApplyCandidate  # noqa: E402
from src.tasks.artifact_gate import ArtifactGate  # noqa: E402
from src.tasks.merge_kernel import MergeKernel  # noqa: E402
from src.tasks.score_perf import ScorePerf  # noqa: E402
from src.tasks.write_opt_report import WriteOptReport  # noqa: E402
from src.workflows.analyze_flow import analyze_flow  # noqa: E402
from src.workflows.simulate_flow import simulate_flow  # noqa: E402
from wfpy import File, connect, workflow  # noqa: E402


@workflow(
    inputs={"Kernel": str},
    outputs={"Report": File(ext=".md"), "Advice": str},
)
def optimize_flow() -> None:
    """Measure, decide, edit, re-measure -- with a gate and a doctor."""
    merge = MergeKernel()
    simulate = simulate_flow()
    analyze = analyze_flow()
    score = ScorePerf(budget=BUDGET)
    planner = TunePlanner()
    gate = ArtifactGate()
    applier = TuneApplier()
    doctor = TileDoctor()
    candidate = ApplyCandidate()
    writer = WriteOptReport()
    advisor = PerfAdvisor()

    # Measure: the two phases sim_analyze uses, unchanged.
    connect("Kernel", merge.Initial)
    connect(merge.Out, simulate.Kernel)
    connect(simulate.Result, analyze.Result)
    connect(analyze.Perf, score.Perf)

    # Decide, then edit: plan -> verify the plan -> apply exactly it.
    connect(score.Request, planner.Request)
    connect(planner.Plan, gate.Plan)
    # The gate needs the file, not just the plan: a plan says what to change,
    # and the applier is the node that has to produce the changed file.
    connect(score.Source, gate.Source)
    connect(gate.Directive, applier.Directive)
    connect(applier.Candidate, candidate.Candidate)

    # A candidate the compiler rejected goes to the doctor, not to the bin.
    connect(score.Broken, doctor.Failure)
    connect(doctor.Fixed, candidate.Fixed)

    # Either way, the next kernel goes back round.
    connect(candidate.Kernel, merge.Retry)

    # Exit, and then explain the result.
    connect(score.Result, writer.Result)
    connect(writer.Report, "Report")
    connect(writer.Analysis, advisor.Analysis)
    connect(advisor.Advice, "Advice")
