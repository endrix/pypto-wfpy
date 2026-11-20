#!/usr/bin/env python3
# coding: utf-8
"""Sub-workflow: localise a numeric fault to one op, then explain it.

    "Suspect"   --> verify_flow(suspect)   --+
                                             |--> CompareDumps --> WriteTriageReport
    "Reference" --> verify_flow(reference) --+                           |
                                                                   Report | Brief | Verdict
                                                                          v
                                                                   PerfAnalyst --> "Analysis"

Five nodes, two of them the same sub-workflow instantiated twice -- once per
kernel -- which is the whole trick: one `verify_flow` would have the two runs
fighting over the same work directory.

The division of labour is the point. `CompareDumps` is **arithmetic**: it aligns
the two runs' dumps by op magic, which is stable across runs, and compares them
elementwise, so naming the earliest divergent op is a fact and not an opinion.
`PrecisionAnalyst` is **judgement**: mapping a tile-level opcode back to a source
construct, since one `pypto.softmax` lowers to five primitives. No agent takes
part in the localisation, and the localisation is not asked to explain itself.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.precision_analyst import PrecisionAnalyst  # noqa: E402
from src.tasks.compare_dumps import CompareDumps  # noqa: E402
from src.tasks.write_triage_report import WriteTriageReport  # noqa: E402
from src.workflows.verify_flow import verify_flow  # noqa: E402
from wfpy import File, connect, workflow  # noqa: E402


@workflow(
    inputs={"Suspect": str, "Reference": str},
    outputs={"Report": File(ext=".md"), "Analysis": File(ext=".md"),
             "Verdict": str},
)
def triage_flow() -> None:
    """Two instrumented runs, an exact comparison, and one explanation."""
    # Factory form, and different slots: see verify_flow for why both matter.
    suspect = verify_flow("suspect", slot=0)()
    reference = verify_flow("reference", slot=1)()
    compare = CompareDumps()
    writer = WriteTriageReport()
    analyst = PrecisionAnalyst()

    connect("Suspect", suspect.Kernel)
    connect("Reference", reference.Kernel)

    connect(suspect.Dumps, compare.Suspect)
    connect(reference.Dumps, compare.Reference)

    connect(compare.Divergence, writer.Divergence)
    connect(writer.Report, "Report")
    connect(writer.Verdict, "Verdict")

    connect(writer.Brief, analyst.Brief)
    connect(analyst.Analysis, "Analysis")
