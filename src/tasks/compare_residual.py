"""Evaluate both independent designs on identical tensors on one NPU."""

import json
from pathlib import Path
from statistics import median

from src.lib.residual_contract import BRANCHES, HELD_OUT_SHAPE, TIMING_SHAPE, REFERENCE
from src.lib.residual_run import run_comparison
from wfpy import File, Port, action, task


@task
class CompareResidual:
    workspace: str
    device: int

    class Ports:
        Row = Port[File](direction="in")
        Split = Port[File](direction="in")
        Report = Port[File](direction="out", ext=".md")

    @action(consumes={"Row": 1, "Split": 1}, produces={"Report": 1})
    def compare(self, row, split):
        records = {name: json.loads(Path(str(token)).read_text())
                   for name, token in zip(BRANCHES, (row, split))}
        kernels = {name: r["kernel"] for name, r in records.items() if r["status"] == "PASS"}
        workspace = Path(self.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        details = ""
        if kernels:
            result, directory, error = run_comparison(kernels, workspace, self.device)
            details = f"Hardware evidence and samples: {directory}"
        else:
            result, error = {}, "Neither branch produced a verified candidate."
        failures = result.get("errors", {})
        complete = len(kernels) == 2 and not failures and not error
        lines = ["# Residual RMSNorm comparison", "", "**COMPLETE**" if complete else "**INCOMPLETE**",
                 "", f"NPU {self.device}. Training timing shape {TIMING_SHAPE}, held-out shape {HELD_OUT_SHAPE}.",
                 "", f"Unfused PyTorch baseline: {REFERENCE}", "", "## Branches", ""]
        for name, record in records.items():
            lines += [f"- {name}: {record['status']}. Kernel: {record['kernel']}. "
                      f"Branch report: {record['report']}"]
        if error:
            lines += ["", error]
        for name, message in failures.items():
            lines += ["", f"{name} failed final verification: {message}"]
        if result:
            lines += ["", f"Selected among verified methods: **{result['winner']}**.", "",
                      "| Method | Shape | Median synchronized latency (ms) |",
                      "|---|---|---:|"]
            for name, by_shape in result["samples_ms"].items():
                for shape, values in by_shape.items():
                    lines.append(f"| {name} | {shape} | {median(values):.3f} |")
            lines += ["", "After three warmups, each method ran twelve times with rotating order. "
                      "To replace unfused PyTorch a candidate must win nine rounds and improve median "
                      "latency by at least 2% on each shape. Eligible branches are ranked by their "
                      "geometric mean latency ratio."]
        report = workspace / "comparison.md"
        report.write_text("\n".join(lines + ["", details, ""]))
        return str(report)
