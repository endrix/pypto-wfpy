#!/usr/bin/env python3
# coding: utf-8
"""Node: `ScorePerf` -- the optimization loop's only exit."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
import sys

from typing import Any

from src.lib import bubbles
from src.lib.kda_eval import diagnostic
from src.lib.optimize_config import MIN_GAIN_PCT, REPAIR_BUDGET, TUNE_ROOT
from src.lib.records import PerfData, SearchResult
from wfpy import Port, action, guard, task


@task
class ScorePerf:
    """Compare a candidate against the incumbent, then ask, accept or stop.

    The objective is total simulated cycles: lower wins. The simulator is
    deterministic, so a difference is a real difference in the cost model --
    but it is a MODEL, so a gain here is a prediction, not a measurement.

    Three mutually exclusive guards, so exactly one action can fire. wfpy picks
    nondeterministically among actions that are all activated -- declaration
    order happens to break the tie in this runtime, but the model does not
    promise it -- so each guard states its full condition.

    A broken candidate is not a dead end: it scores as infinitely slow, the
    incumbent stands, and the budget pays for the attempt. That is why the loop
    terminates even when the agent returns nonsense, which is the whole reason
    the budget is charged for a failed candidate.
    """

    budget: int

    repair_budget: int = REPAIR_BUDGET  # has a default -> state, not a parameter

    _rounds: int = 0
    _repairs: int = 0
    _best_cycles: float = 0.0
    _best_kernel: str = ""
    _best_perf: Any = None  # PerfData of the incumbent, for the request
    _baseline_cycles: float = 0.0
    # A string, not a list: an annotated field with a default becomes per-actor
    # state, and a mutable default would be one object shared by every instance.
    _history: str = ""

    class Ports:
        Perf = Port[PerfData](direction="in")
        Request = Port[str](direction="out")
        # The incumbent's path, emitted with every Request. The planner is given
        # the source inline in the request, but its PLAN does not contain the
        # source, so without this the applier downstream is asked to rewrite a
        # file it was never shown -- measured, it replied
        # `{"outputs": {"Candidate": ""}}`, which was the correct answer to the
        # question actually asked.
        Source = Port[str](direction="out")
        Broken = Port[str](direction="out")
        Result = Port[SearchResult](direction="out")

    # ---- routing ---------------------------------------------------------

    @staticmethod
    def _ran(perf: PerfData) -> bool:
        """Whether this run produced usable performance data at all."""
        return perf.result.passed and perf.found and 0 < perf.total_cycles and isfinite(perf.total_cycles)

    def _route(self, perf: PerfData) -> str:
        """Which single exit this measurement takes.

        One pure function rather than four independent guard lambdas. wfpy picks
        nondeterministically among actions whose guards all pass, so the exits
        have to be mutually exclusive; deriving them from one routing decision
        makes that structural instead of something four conditions have to agree
        on by hand. It mutates nothing -- a guard may be evaluated more than once.
        """
        first = not self._best_kernel
        ran = self._ran(perf)
        rounds_left = self._rounds < self.budget
        repairs_left = self._repairs < self.repair_budget

        if first and not ran:
            # Nothing to optimize: the kernel was already broken or never
            # produced cost-model data (a device run, for instance).
            return "diagnose" if perf.result.job.phase and repairs_left else "abort"
        if not first and not ran:
            # A candidate the compiler or the self-check rejected. That is a
            # constraint problem, not a slower kernel, and it has its own agent.
            if repairs_left and (rounds_left or (perf.result.job.phase and perf.result.rc == -2)):
                return "diagnose"
            if rounds_left:
                # Out of repairs for THIS lever. The lever is not the search:
                # ask for a different catalogue item rather than ending while
                # items remain untried.
                return "ask"
            return "stop"
        if ran and rounds_left:
            return "ask"
        return "stop"

    # ---- scoring -------------------------------------------------------

    @classmethod
    def _cycles(cls, perf: PerfData) -> float:
        """The objective. A run that did not produce data cannot win."""
        return perf.total_cycles if cls._ran(perf) else float("inf")

    def _record(self, perf: PerfData) -> tuple[float, bool]:
        """Score this run, update the incumbent, and say whether it won."""
        cycles = self._cycles(perf)
        kernel = perf.result.job.kernel
        if not self._best_kernel or (isfinite(cycles) and not isfinite(self._best_cycles)):
            # First run: it is the incumbent by default and the baseline every
            # later gain is quoted against.
            self._best_cycles, self._best_kernel = cycles, kernel
            self._best_perf = perf
            self._baseline_cycles = cycles
            self._note(kernel, cycles, "baseline")
            return cycles, True

        gain = (
            100.0 * (self._best_cycles - cycles) / self._best_cycles
            if self._best_cycles not in (0.0, float("inf"))
            else 0.0
        )
        won = cycles < self._best_cycles and gain > MIN_GAIN_PCT
        if won:
            self._best_cycles, self._best_kernel = cycles, kernel
            self._best_perf = perf
        # A candidate with no data has no gain to quote -- "-inf%" is noise.
        why = "rejected (did not run)" if cycles == float("inf") else (
            f"{'adopted' if won else 'rejected'} ({gain:+.1f}%)"
        )
        if not self._ran(perf):
            why += f" {perf.result.error or 'no cycle report'}; logs: {perf.result.job.work_dir}"
        self._note(kernel, cycles, why)
        return cycles, won

    def _note(self, kernel: str, cycles: float, note: str) -> None:
        shown = "no data" if cycles == float("inf") else f"{cycles:,.0f}"
        change = Path(kernel).with_suffix(".change.txt")
        if change.is_file():
            note += " " + change.read_text(encoding="utf-8")[:600]
        self._history += f"      {Path(kernel).name:<22} {shown:>12}  {note}\n"

    def _log(self, perf: PerfData, cycles: float, won: bool, final: bool = False) -> None:
        name = perf.result.job.name
        shown = "no data" if cycles == float("inf") else f"{cycles:,.0f} cycles"
        # Neither the final measurement nor a repair is a new round: the first
        # consumed no budget, the second consumed the repair budget. Labelling
        # either with the round number reads as a repeat of that round.
        where = "final" if final else f"round {self._rounds}"
        print(
            f"[opt] {where}: {name} -> {shown} "
            f"{'ADOPTED' if won else 'kept incumbent'} "
            f"(best {self._best_cycles:,.0f})",
            file=sys.stderr,
        )

    # ---- the request the agent gets -------------------------------------

    def _request(self, measured: PerfData) -> str:
        """Everything the optimizer needs, in one token.

        Composed here rather than joined at the agent: the agent fires on its
        inputs, so pairing separately-arriving advice, metrics and source would
        need a join node that only concatenates strings. This node already
        holds all three.
        """
        try:
            source = Path(self._best_kernel).read_text(encoding="utf-8")
        except OSError:
            source = "(source unavailable)"

        # The INCUMBENT's measurement, never the caller's: this path is also
        # reached after a candidate was rejected, and that candidate's empty
        # counters would be quoted beside the incumbent's source.
        perf = self._best_perf if self._best_perf is not None else measured
        ranked = sorted(perf.pipe_pct.items(), key=lambda kv: kv[1], reverse=True)
        pipes = ", ".join(f"{p}={v}%" for p, v in ranked[:5]) or "(none)"
        data = bubbles.parse(Path(perf.sim_dir or ".") / "bubble_analysis.log")
        cores = (
            f"{len(data.busy)} of {len(data.compute_cores)} compute cores ran "
            f"({data.occupancy_pct}%); fleet utilization {data.mean_utilization_pct}%; "
            f"mean stall share on busy cores {data.mean_bubble_pct}%"
            if data.found
            else "no core data"
        )
        waits = "unknown"
        if data.found and data.busy:
            schedule = sum(c.wait_schedule for c in data.busy)
            predecessor = sum(c.wait_predecessor for c in data.busy)
            if schedule or predecessor:
                waits = "schedule" if schedule >= predecessor else "predecessor"
        fixture = "128x64 FP32, seed=0, diagonal=1, cumulative gates" if perf.result.job.phase else "kernel self-check"
        catalogue = TUNE_ROOT / "shared" / "optimization_catalog.md"

        return (
            f"Round {self._rounds} of {self.budget}.\n"
            f"Incumbent: {self._best_kernel}\n"
            f"Incumbent cost: {self._best_cycles:,.0f} simulated cycles\n"
            f"Case: {fixture}\nMeasurement: {perf.result.job.work_dir}\n"
            f"Component cycles: {perf.component_cycles}; counter source: {perf.sim_dir}\n"
            f"Optimization catalogue: {catalogue}\nPrior trials and outcomes:\n{self._history}\n"
            f"Choose a distinct edit; retain the evaluator entry point and fixture.\n"
            f"Pipe usage: {pipes}\n"
            f"Cores: {cores}\n"
            f"Waiting is dominated by: {waits}\n"
            f"Target: beat {self._best_cycles:,.0f} cycles by more than "
            f"{MIN_GAIN_PCT}% without failing the self-check.\n\n"
            f"--- current source ---\n{source}"
        )

    def _failure(self, perf: PerfData) -> str:
        """The rejection, with the source that caused it.

        The error text is the point. Scoring a rejected candidate as infinitely
        slow and discarding its diagnostic is what let a measured INVALID_TILE
        be repeated by the next round.
        """
        job = perf.result.job
        try:
            source = Path(job.kernel).read_text(encoding="utf-8")
        except OSError:
            source = "(source unavailable)"
        return (
            f"Repair {self._repairs} of {self.repair_budget}.\n"
            f"Rejected candidate: {job.kernel}\n"
            f"Exit code: {perf.result.rc}\n"
            f"{diagnostic(perf.result)}\n"
            "Preserve the operator interface and fixtures. Diagnose the recorded phase before editing.\n"
            f"Incumbent to beat: {self._best_cycles:,.0f} simulated cycles\n\n"
            f"--- rejected source ---\n{source}"
        )

    # ---- the verdict the report gets ------------------------------------

    def _verdict(self, tag: str) -> str:
        head = (f"{tag}  best={self._best_cycles:,.0f} cycles  kernel={self._best_kernel}"
                if self._best_kernel else f"{tag}  no scored kernel")
        lines = [head]
        if self._baseline_cycles not in (0.0, float("inf")) and self._best_cycles < self._baseline_cycles:
            gain = 100.0 * (self._baseline_cycles - self._best_cycles) / self._baseline_cycles
            lines.append(f"      {gain:.1f}% below the baseline of {self._baseline_cycles:,.0f}")
        return "\n".join(lines) + ("\n" + self._history.rstrip() if self._history else "")

    def _finish(self, tag: str, perf: PerfData) -> SearchResult:
        winner = self._best_perf or perf
        result = SearchResult(self._verdict(tag), winner.result.job.work_dir, self._best_kernel)
        # wfpy reuses the nested actor when the outer graph starts a new design.
        self._rounds = self._repairs = 0
        self._best_cycles = self._baseline_cycles = 0.0
        self._best_perf = None
        self._best_kernel = self._history = ""
        return result

    # ---- actions --------------------------------------------------------

    @action(consumes={"Perf": 1}, produces={"Request": 1, "Source": 1})
    @guard(lambda self, perf: self._route(perf) == "ask")
    def ask_planner(self, perf: PerfData) -> dict:
        self._rounds += 1
        # Each lever gets its own repair allowance, so one unfixable tiling does
        # not spend the next lever's. Total repairs stay bounded at
        # budget * repair_budget.
        self._repairs = 0
        cycles, won = self._record(perf)
        self._log(perf, cycles, won)
        return {"Request": self._request(perf), "Source": self._best_kernel}

    @action(consumes={"Perf": 1}, produces={"Broken": 1})
    @guard(lambda self, perf: self._route(perf) == "diagnose")
    def diagnose(self, perf: PerfData) -> str:
        self._repairs += 1
        # A rejected candidate does not become the incumbent, but it is recorded
        # so the history shows the round was spent and why.
        self._note(perf.result.job.kernel, float("inf"),
                   f"evaluation failed ({perf.result.error or 'no usable cycle report'}); logs: {perf.result.job.work_dir}")
        print(f"[opt] evaluation failed: {perf.result.error}; repair {self._repairs}/{self.repair_budget}",
              file=sys.stderr)
        return self._failure(perf)

    @action(consumes={"Perf": 1}, produces={"Result": 1})
    @guard(lambda self, perf: self._route(perf) == "abort")
    def abort(self, perf: PerfData) -> SearchResult:
        self._note(perf.result.job.kernel, float("inf"), "baseline unavailable")
        self._history += diagnostic(perf.result) + "\n"
        return self._finish("BASELINE UNUSABLE", perf)

    @action(consumes={"Perf": 1}, produces={"Result": 1})
    @guard(lambda self, perf: self._route(perf) == "stop")
    def stop(self, perf: PerfData) -> SearchResult:
        cycles, won = self._record(perf)
        self._log(perf, cycles, won, final=True)
        # "ADOPTED" only if the last candidate actually won; the tag reports the
        # search as a whole, so it says whether anything beat the baseline.
        improved = (
            self._baseline_cycles not in (0.0, float("inf"))
            and self._best_cycles < self._baseline_cycles
        )
        return self._finish("OPTIMIZED" if improved else "NO GAIN", perf)
