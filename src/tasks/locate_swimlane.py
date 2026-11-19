#!/usr/bin/env python3
# coding: utf-8
"""Node: `LocateSwimlane` -- hand the swimlane trace to a viewer."""

from __future__ import annotations

import json
from pathlib import Path

from src.lib.records import PerfData
from wfpy import File, Port, action, task

# The simulator writes several traces; this is the one the PyPTO Toolkit's
# swimlane view reads (its handler also accepts a `perf_swimlane.json`, and
# falls back to treating the file as a raw Chrome `traceEvents` dump).
SWIMLANE = "merged_swimlane.json"


@task
class LocateSwimlane:
    """Find the simulator's merged swimlane trace for one run.

    Its own node rather than a branch inside the perf analysis: the swimlane is
    not summarised here at all, only located and passed on, because what reads
    it is the Toolkit's renderer and not this workflow.
    """

    class Ports:
        Perf = Port[PerfData](direction="in")
        Swimlane = Port[File](direction="out", ext=".json")

    @action(consumes={"Perf": 1}, produces={"Swimlane": 1})
    def locate(self, perf: PerfData) -> str:
        found = Path(perf.sim_dir) / SWIMLANE if perf.sim_dir else None
        if found and found.is_file():
            return str(found)

        # The port is an unconditional output, so something has to go down it.
        # A stub in the trace's own shape is the least confusing choice: the
        # Toolkit opens it and says there is no valid data, rather than failing
        # to parse a markdown file handed to a trace renderer.
        stub = Path(perf.result.job.work_dir) / "swimlane_missing.json"
        stub.write_text(
            json.dumps(
                {
                    "traceEvents": [],
                    "note": (
                        f"No {SWIMLANE} under {perf.sim_dir or '(no simulator output dir)'}. "
                        "The cost model writes it in sim mode only."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(stub)
