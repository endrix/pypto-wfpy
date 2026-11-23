#!/usr/bin/env python3
# coding: utf-8
"""Sub-workflow factory: profile one kernel on the device.

    "Kernel" --> InstrumentDebug --> RunProfile --> "Profile"

cannbot's S2_COLLECT. A **factory**, like `verify_flow`, because a `@workflow`
builder is called by wfpy with no arguments, so parameters given at the call
site are ignored -- and both arguments here matter. `label` keeps two kernels'
instrumented copies and work directories apart; `slot` decides the device, since
two profiling runs on one NPU interfere and the symptom is wrong numbers rather
than an error.

The swimlane and in-core tuning phases each have to re-collect on entry -- the
skills forbid reusing another phase's data, because the code has changed since.
So this is instantiated per phase rather than once per run.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lib.records import Measurement  # noqa: E402
from src.tasks.instrument_debug import InstrumentDebug  # noqa: E402
from src.tasks.run_profile import RunProfile  # noqa: E402
from wfpy import connect, workflow  # noqa: E402


def collect_flow(label: str, slot: int) -> Callable[[], None]:
    """Build a profiling workflow for one kernel."""

    @workflow(inputs={"Kernel": str}, outputs={"Profile": Measurement})
    def flow() -> None:
        """Instrument for profiling, run on the device, hand back the data."""
        instrument = InstrumentDebug(label=label)
        runner = RunProfile(label=label, slot=slot)

        connect("Kernel", instrument.Kernel)
        connect(instrument.Instrumented, runner.Instrumented)
        connect(runner.Profile, "Profile")

    return flow
