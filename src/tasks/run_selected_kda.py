"""Run the simulator-selected kernel on the NPU before publishing the search result."""

from dataclasses import replace
from pathlib import Path

from src.lib.device_tune import compare_kda_pair
from src.lib.kda_eval import diagnostic, evaluate
from src.lib.records import SearchResult
from wfpy import Port, action, task


@task
class RunSelectedKda:
    workspace: str
    op: str

    class Ports:
        Selected = Port[SearchResult](direction="in")
        Checked = Port[SearchResult](direction="out")

    @action(consumes={"Selected": 1}, produces={"Checked": 1})
    def check(self, selection: SearchResult) -> SearchResult:
        if not selection.kernel:
            return selection
        run = evaluate("final", selection.kernel, self.workspace, self.op)
        if not run.passed:
            verdict = f"NPU FAILED  kernel={selection.kernel}\nSimulator selection: {selection.verdict}"
            return replace(selection, verdict=verdict, npu_report=diagnostic(run))
        comparison = "The baseline was the only simulator finalist."
        baseline = selection.baseline_kernel
        if baseline and baseline != selection.kernel:
            before = evaluate("final", baseline, self.workspace, self.op)
            if not before.passed:
                return replace(selection, verdict="NPU FAILED  baseline verification", npu_report=diagnostic(before))
            try:
                adopt, evidence = compare_kda_pair(Path(baseline), Path(selection.kernel),
                                                   Path(run.job.work_dir) / "paired", run.job.device_id)
            except (ValueError, OSError) as exc:
                return replace(selection, verdict="NPU FAILED  paired measurement", npu_report=str(exc))
            comparison = (f"Simulator finalist: {selection.kernel}. Paired NPU timing "
                          f"(baseline → finalist): {evidence}")
            if not adopt:
                verdict = (f"NO GAIN  best={selection.baseline_cycles:,.0f} cycles  kernel={baseline}\n"
                           f"Simulator selection: {selection.verdict}")
                selection = replace(selection, kernel=baseline, measurement_dir=selection.baseline_dir,
                                    verdict=verdict)
                run = before
        latency = next(line for line in run.stdout.splitlines() if line.startswith("NPU latency:"))
        report = (f"Passed all five correctness fixtures on NPU {run.job.device_id}.\n\n"
                  f"{latency}\n\n{comparison}\n\nSelected kernel: `{selection.kernel}`\n\n"
                  f"Execution log: `{run.job.work_dir}/stdout.log`")
        return replace(selection, npu_report=report)
