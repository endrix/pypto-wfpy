#!/usr/bin/env python3
# coding: utf-8
"""Token types passed between actors.

These are frozen dataclasses, not dicts, on purpose. With a *single* declared
output port wfpy reinterprets a returned dict as a {port_name: value} mapping
(wfpy/_action_runtime.py:269), so returning a dict payload emits nothing at all,
silently. A dataclass passes through untouched.

Data types are allowed to share a module; the one-per-file rule is about actors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JobSpec:
    """What to run, and where."""

    kernel: str
    name: str
    exists: bool
    run_mode: str
    work_dir: str
    cmd: tuple[str, ...]
    slot: int = 0
    device_id: int = 0


@dataclass(frozen=True)
class RunResult:
    """How the run went."""

    job: JobSpec
    rc: int
    stdout: str
    error: str
    seconds: float

    @property
    def passed(self) -> bool:
        # Both conditions matter: exit 0 alone would also accept a kernel that
        # printed nothing, and a PASS line alone would accept a late crash.
        return self.rc == 0 and "PASS:" in self.stdout

    @property
    def sim_output_dir(self) -> str:
        """Newest simulator output dir for this run, or "" if there is none."""
        outputs = [p for p in Path(self.job.work_dir).glob("output/*") if p.is_dir()]
        if not outputs:
            return ""
        return str(max(outputs, key=lambda p: p.stat().st_mtime))


@dataclass(frozen=True)
class Outcome:
    """One repair attempt: what it computed, and what the compiler made of it."""

    result: RunResult
    graph: Any           # graph.GraphSummary; untyped here to avoid a cycle
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


@dataclass(frozen=True)
class IRRun:
    """A run made with per-pass IR dumping enabled."""

    job: JobSpec
    rc: int
    error: str
    seconds: float
    dump_root: str  # the computation_graph dir, or "" if absent


@dataclass(frozen=True)
class PassDiff:
    """One compiler pass, for one function, before vs after."""

    strategy: str
    index: int
    pass_name: str
    function: str
    level_before: str
    level_after: str
    added: int
    removed: int
    diff_text: str

    @property
    def changed(self) -> bool:
        return self.added > 0 or self.removed > 0

    @property
    def level_shift(self) -> bool:
        return self.level_before != self.level_after


@dataclass(frozen=True)
class PerfData:
    """Parsed simulator performance counters for one kernel."""

    result: RunResult
    found: bool
    sim_dir: str
    total_cycles: float = 0.0
    pipe_pct: dict = field(default_factory=dict)     # pipe -> usage %
    pipe_cycles: dict = field(default_factory=dict)  # pipe -> avg cycles
    counters: dict = field(default_factory=dict)     # other scalar counters
    tile_ops: dict = field(default_factory=dict)     # op name -> work amount
