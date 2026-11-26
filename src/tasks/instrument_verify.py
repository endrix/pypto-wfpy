#!/usr/bin/env python3
# coding: utf-8
"""Node: `InstrumentVerify` -- write a pass-verifying copy of a kernel."""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

from src.lib import verify
from src.lib.runner import REPO_ROOT
from wfpy import Port, action, task


@task
class InstrumentVerify:
    """Copy a kernel with `verify_options` added, never editing the original.

    A copy, not an in-place edit, for the same reason the repair loop writes its
    candidates beside the input: the kernel under triage is evidence. The copy
    keeps the original's file name under a per-run directory so the dumps of two
    kernels stay apart.
    """

    label: str  # "reference" or "suspect"; names the directory the copy goes in

    class Ports:
        Kernel = Port[str](direction="in")
        Instrumented = Port[str](direction="out")

    @action(consumes={"Kernel": 1}, produces={"Instrumented": 1})
    def instrument(self, kernel: str) -> str:
        # str(): a @source emits its File dataclass, not a path string.
        source_path = Path(str(kernel))
        out_dir = REPO_ROOT / "runs" / "triage" / self.label
        out_dir.mkdir(parents=True, exist_ok=True)
        target = Path(mkdtemp(prefix='run-', dir=out_dir)) / source_path.name
        target.write_text(
            verify.instrument(source_path.read_text(encoding="utf-8")), encoding="utf-8"
        )
        return str(target)
