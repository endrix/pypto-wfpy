"""Run the simulator-selected kernel on the NPU before publishing the search result."""

from dataclasses import replace

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
        latency = next(line for line in run.stdout.splitlines() if line.startswith("NPU latency:"))
        report = (f"Passed all five correctness fixtures on NPU {run.job.device_id}.\n\n"
                  f"{latency}\n\nExecution log: `{run.job.work_dir}/stdout.log`")
        return replace(selection, npu_report=report)
