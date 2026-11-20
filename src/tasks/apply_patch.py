#!/usr/bin/env python3
# coding: utf-8
"""Write the repair agent's reply to a file the runner can execute."""

from __future__ import annotations

from pathlib import Path

from src.lib.agent_io import read_json
from src.lib.runner import REPO_ROOT
from wfpy import Port, action, task


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
        out_dir = REPO_ROOT / "runs" / "repaired"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"attempt{self._n}.py"
        # The agent's port is .json, so the reply is an object with the source
        # in a field. read_json raises on a malformed one rather than writing a
        # half-empty file: an empty kernel reads downstream as "it ran and
        # printed nothing", which is how an empty payload once passed for a
        # successful turn.
        reply = read_json(patched, required=("kernel",))
        target.write_text(reply["kernel"].rstrip("\n") + "\n", encoding="utf-8")
        return str(target)
