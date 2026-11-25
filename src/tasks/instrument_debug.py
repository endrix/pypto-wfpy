#!/usr/bin/env python3
# coding: utf-8
"""Node: `InstrumentDebug` -- write a profiling copy of a kernel."""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

from src.lib import profile
from src.lib.jit_options import add_option
from src.lib.runner import REPO_ROOT
from wfpy import Port, action, task


@task
class InstrumentDebug:
    """Copy a kernel with `debug_options={"runtime_debug_mode": 1}` added.

    A copy, never an edit in place. That is also how the orchestrator's
    requirement to restore the options at the end of the phase is met: there is
    nothing to restore when the original was never touched.
    """

    label: str

    class Ports:
        Kernel = Port[str](direction="in")
        Instrumented = Port[str](direction="out")

    @action(consumes={"Kernel": 1}, produces={"Instrumented": 1})
    def instrument(self, kernel: str) -> str:
        # str(): a @source emits its File dataclass, not a path string.
        source_path = Path(str(kernel))
        out_dir = REPO_ROOT / "runs" / "profile" / self.label
        out_dir.mkdir(parents=True, exist_ok=True)
        target = Path(mkdtemp(prefix=f"{source_path.stem}-", dir=out_dir)) / source_path.name
        target.write_text(
            add_option(
                source_path.read_text(encoding="utf-8"),
                "debug_options",
                profile.DEBUG_OPTIONS,
                note=profile.DEBUG_NOTE,
            ),
            encoding="utf-8",
        )
        return str(target)
