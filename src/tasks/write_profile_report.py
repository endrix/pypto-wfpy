#!/usr/bin/env python3
# coding: utf-8
"""Node: `WriteProfileReport` -- the measured picture, from the vendor readers."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from src.lib import profile
from src.lib.records import Measurement
from src.lib.runner import REPO_ROOT
from wfpy import File, Port, action, task


@task
class WriteProfileReport:
    """Run the tuning skill's own readers and write what they say.

    Not reimplemented: `analyze_swimlane.py` already knows the swimlane's merge
    keys and hash order, and `analyze_core_usage.py` asks the device for its
    real core count. Both are run through the project interpreter with the CANN
    environment -- under a bare `python3` the second one reports "Failed to get
    theoretical cores from torch.npu ... may underestimate" and its numbers are
    quietly wrong.

    These are MEASURED microseconds from a device run. Everything the simulator
    reports is cycles from a cost model. They are not comparable, and only one
    of them is a measurement -- so the report says which it is.
    """

    class Ports:
        Profile = Port[Measurement](direction="in")
        Report = Port[File](direction="out", ext=".md")
        Verdict = Port[str](direction="out")

    @action(consumes={"Profile": 1}, produces={"Report": 1, "Verdict": 1})
    def write(self, run: Measurement) -> dict:
        out_dir = REPO_ROOT / "runs" / "profile"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = Path(run.job.work_dir) / f"{run.job.name}_profile.md"
        latest = out_dir / path.name

        def publish(document: str) -> None:
            path.write_text(document, encoding="utf-8")
            # Existing callers use this index. Publish a whole report atomically.
            with NamedTemporaryFile(mode="w", dir=out_dir, encoding="utf-8", delete=False) as tmp:
                tmp.write(document)
            Path(tmp.name).replace(latest)

        if not run.found:
            verdict = (
                f"NO DATA  {run.job.name}: the run produced no swimlane data"
                + (f" ({run.error})" if run.error else "")
            )
            publish(
                f"# Device profile -- {run.job.name}\n\n**{verdict}**\n\n"
                "A profiling run needs the NPU: the data is copied off the\n"
                "device, so a sim run has nothing to copy.\n",
            )
            return {"Report": str(path), "Verdict": verdict}

        swimlane = profile.swimlane_report(run.profile_dir)
        cores = profile.core_usage_report(run.profile_dir, device_id=run.job.device_id)

        # The vendor readers print torch_npu's import warnings alongside their
        # report; they are noise in a document meant to be read.
        def clean(text: str) -> str:
            return "\n".join(
                line for line in text.splitlines()
                if "site-packages" not in line and "warnings.warn" not in line
            ).strip()

        verdict = f"PROFILED  {run.job.name}: {'passed' if run.passed else 'FAILED its own check'}"
        lines = [
            f"# Device profile -- {run.job.name}", "",
            f"**{verdict}**", "",
            f"- kernel: `{run.job.kernel}`",
            f"- profile invocation: `{' '.join(run.job.cmd)}`",
            f"- instrumented SHA-256: `{run.instrumented_sha256}`",
            f"- device: {run.job.device_id}",
            f"- data: `{run.profile_dir}`",
        ]
        if run.workspace_kb:
            lines.append(
                f"- the run recommends `max_workspace_kb` near **{run.workspace_kb}** "
                "(catalogue item S-20; set it in `runtime_options` to activate "
                "memory-driven mode)"
            )
        lines += [
            "",
            "## Core usage, against the hardware's real core count", "",
            "```", clean(cores) or "(no output)", "```",
            "",
            "## Per-subgraph timings and merge guidance", "",
            "```", clean(swimlane) or "(no output)", "```",
            "",
            "## Reading this", "",
            "These are **measured microseconds** from a run on the NPU. The",
            "simulator's reports elsewhere in this project are cycles from a cost",
            "model: the two are not comparable, and only this one is a measurement.",
            "",
            "`FILL CORES FIRST` is catalogue item S-2, and the catalogue puts S-1",
            "(core usage analysis) before it as a prerequisite -- which is what this",
            "report is.",
        ]
        publish("\n".join(lines) + "\n")
        return {"Report": str(path), "Verdict": verdict}
