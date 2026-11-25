#!/usr/bin/env python3
# coding: utf-8
"""Node: `WriteOptReport` -- the search's verdict, and the advisor's question."""

from __future__ import annotations

from pathlib import Path

from src.lib.records import SearchResult
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

    directory: str

    class Ports:
        Result = Port[SearchResult](direction="in")
        Report = Port[File](direction="out", ext=".md")
        Analysis = Port[str](direction="out")

    @action(consumes={"Result": 1}, produces={"Report": 1, "Analysis": 1})
    def write(self, result: SearchResult) -> dict:
        out_dir = Path(self.directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        verdict, _, history = result.verdict.partition("\n")
        perf_md, bubble_md = self._collect(Path(result.measurement_dir))

        lines = [
            "# Optimization search",
            "",
            f"**{verdict.strip()}**",
            "",
            *(["## Final NPU execution", "", result.npu_report, ""] if result.npu_report else []),
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
            "Cycle counts and cycle reductions are simulator estimates.",
        ]
        path = out_dir / "search.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        analysis = (
            f"{verdict.strip()}\n\nCandidates tried:\n{history.rstrip()}\n\n"
            f"Measurement directory: {result.measurement_dir}\n{perf_md}\n\n{bubble_md}\n{result.npu_report}"
        )
        return {"Report": str(path), "Analysis": analysis}

    @staticmethod
    def _collect(run_dir: Path) -> tuple[str, str]:
        """Read the reports from the specific winning evaluation."""
        perf_path, bubble_path = run_dir / "perf_report.md", run_dir / "bubble_report.md"
        perf = perf_path.read_text(encoding="utf-8") if perf_path.is_file() else ""
        bubble = bubble_path.read_text(encoding="utf-8") if bubble_path.is_file() else ""
        return perf, bubble
