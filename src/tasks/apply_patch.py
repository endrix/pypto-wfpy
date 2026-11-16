#!/usr/bin/env python3
# coding: utf-8
"""Write the repair agent's reply to a file the runner can execute."""

from __future__ import annotations

from pathlib import Path

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
        text = patched.strip()
        if text.startswith("```"):  # strip a fence the agent added despite the prompt
            lines = text.splitlines()
            text = "\n".join(lines[1 : -1 if lines[-1].strip() == "```" else len(lines)])
        target.write_text(text + "\n", encoding="utf-8")
        return str(target)
