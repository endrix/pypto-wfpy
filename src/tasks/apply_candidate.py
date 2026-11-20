#!/usr/bin/env python3
# coding: utf-8
"""Node: `ApplyCandidate` -- write the optimizer's reply to a runnable file."""

from __future__ import annotations

from pathlib import Path

from src.lib.agent_io import read_json
from src.lib.runner import REPO_ROOT
from wfpy import Port, action, task


@task
class ApplyCandidate:
    """Write each candidate beside the original, never over it.

    Accepts an edit from the applier or a correction from the doctor, and
    numbers them in one series so `runs/candidates/` reads in the order tried.

    Separate from the repair loop's `ApplyPatch` because the two write to
    different directories and number their attempts independently; sharing one
    actor would make a repair attempt and an optimization candidate overwrite
    each other's files when both loops run in one process.
    """

    _n: int = 0

    class Ports:
        Candidate = Port[str](direction="in")
        Fixed = Port[str](direction="in")
        Kernel = Port[str](direction="out")

    def _write(self, candidate: str) -> str:
        self._n += 1
        out_dir = REPO_ROOT / "runs" / "candidates"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"candidate{self._n}.py"
        # The agent's port is .json, so the reply is an object with the source
        # in a field. read_json raises on a malformed one rather than writing a
        # half-empty file: an empty kernel reads downstream as "it ran and
        # printed nothing", which is how an empty payload once passed for a
        # successful turn.
        reply = read_json(candidate, required=("kernel",))
        target.write_text(reply["kernel"].rstrip("\n") + "\n", encoding="utf-8")
        return str(target)

    @action(consumes={"Candidate": 1}, produces={"Kernel": 1})
    def apply(self, candidate: str) -> str:
        return self._write(candidate)

    @action(consumes={"Fixed": 1}, produces={"Kernel": 1})
    def apply_fix(self, fixed: str) -> str:
        # Two input ports rather than one, for the same reason MergeKernel has
        # them: a wfpy input port does not fan in, so the applier's edit and the
        # doctor's correction cannot share a port. Both land in the same
        # numbered series, so the candidate files stay in the order tried.
        return self._write(fixed)
