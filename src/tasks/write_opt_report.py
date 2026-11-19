#!/usr/bin/env python3
# coding: utf-8
"""Node: `WriteOptReport` -- the search's verdict, and the advisor's question."""

from __future__ import annotations

from pathlib import Path

from src.lib.runner import REPO_ROOT
from wfpy import File, Port, action, task


@task
class WriteOptReport:
    """Write what the search found, and hand the same facts to the advisor.

    Two outputs from one firing because they are the same content read two
    ways: `Report` is the file a person opens, `Analysis` is the prompt the
    advisor agent answers. Deriving the prompt here keeps the agent off the
    loop's critical path -- it is asked once, at the end, about a result that
    is already final.
    """

    class Ports:
        Result = Port[str](direction="in")
        Report = Port[File](direction="out", ext=".md")
        Analysis = Port[str](direction="out")

    @action(consumes={"Result": 1}, produces={"Report": 1, "Analysis": 1})
    def write(self, result: str) -> dict:
        out_dir = REPO_ROOT / "runs" / "opt_reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        verdict, _, history = result.partition("\n")
        perf_md, bubble_md = self._collect(verdict)

        lines = [
            "# Optimization search",
            "",
            f"**{verdict.strip()}**",
            "",
            "## Candidates tried",
            "",
            "```",
            history.rstrip() or "(only the baseline ran)",
            "```",
            "",
            "## What the cost model said about the winner",
            "",
            perf_md or "_no pipe report found_",
            "",
            bubble_md or "_no core report found_",
            "",
            "## Caveat",
            "",
            "Every number here is the Soc CAModel's estimate, produced on a CPU.",
            "A gain in this report is a predicted gain. It is not a measurement,",
            "and it is not confirmed until the kernel is timed on an NPU.",
        ]
        path = out_dir / "search.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        analysis = (
            f"{verdict.strip()}\n\nCandidates tried:\n{history.rstrip()}\n\n"
            f"{perf_md}\n\n{bubble_md}"
        )
        return {"Report": str(path), "Analysis": analysis}

    @staticmethod
    def _collect(verdict: str) -> tuple[str, str]:
        """Pull in the WINNING kernel's own two reports, if they are on disk.

        Keyed off the kernel named in the verdict, not off the newest `*_sim`
        directory: the newest one belongs to the last candidate tried, which is
        usually the one that lost, so an mtime scan files the loser's numbers
        under "the winner". The analyse phase writes both reports into the work
        dir `make_job` derives from the kernel name, so the winner's directory
        is `runs/<stem>_sim`.

        A missing file is normal -- a candidate that never compiled has no
        reports -- so both halves are optional and the report says so.
        """
        _, _, kernel = verdict.partition("kernel=")
        stem = Path(kernel.strip()).stem
        if not stem:
            return "", ""
        run_dir = REPO_ROOT / "runs" / f"{stem}_sim"
        perf_path, bubble_path = run_dir / "perf_report.md", run_dir / "bubble_report.md"
        perf = perf_path.read_text(encoding="utf-8") if perf_path.is_file() else ""
        bubble = bubble_path.read_text(encoding="utf-8") if bubble_path.is_file() else ""
        return perf, bubble
