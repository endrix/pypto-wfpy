#!/usr/bin/env python3
# coding: utf-8
"""Sub-workflow factory: run one kernel with pass verification on.

    "Kernel" --> InstrumentVerify --> RunVerify --> "Dumps"

A **factory** that returns a workflow, used as `verify_flow("suspect", 0)()`.
A `@workflow` builder is called by wfpy with no arguments, so parameters given
at the call site are ignored -- measured: two instances built with
`label="suspect"` and `label="reference"` both came out labelled with the
default. Closing over the arguments in a factory is the pattern wfpy documents
(`x = factory(...)()`), and it works.

Both arguments matter, and neither is cosmetic:

- `label` keeps the two kernels' instrumented copies and work directories apart.
- `slot` decides the device, via `DEVICES[slot % len(DEVICES)]`. The two branches
  of a triage are independent, so wfpy runs them at the same time; with the same
  slot they land on the same NPU and interfere. Measured: the reference kernel
  failed its own numeric check inside the workflow while passing standalone,
  because both runs were on device 0.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lib.records import DumpSet  # noqa: E402
from src.tasks.instrument_verify import InstrumentVerify  # noqa: E402
from src.tasks.run_verify import RunVerify  # noqa: E402
from wfpy import connect, workflow  # noqa: E402


def verify_flow(label: str, slot: int) -> Callable[[], None]:
    """Build a verification workflow for one kernel."""

    @workflow(inputs={"Kernel": str}, outputs={"Dumps": DumpSet})
    def flow() -> None:
        """Instrument the kernel, run it, hand back what every op dumped."""
        instrument = InstrumentVerify(label=label)
        runner = RunVerify(label=label, slot=slot)

        connect("Kernel", instrument.Kernel)
        connect(instrument.Instrumented, runner.Instrumented)
        connect(runner.Dumps, "Dumps")

    return flow
