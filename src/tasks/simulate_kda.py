"""Run the smoke fixtures before scoring a KDA candidate."""

from src.lib.kda_eval import evaluate
from src.lib.records import RunResult
from wfpy import Port, action, task


@task
class SimulateKda:
    workspace: str
    op: str

    class Ports:
        Kernel = Port[str](direction="in")
        Result = Port[RunResult](direction="out")

    @action(consumes={"Kernel": 1}, produces={"Result": 1})
    def simulate(self, kernel: str) -> RunResult:
        smoke = evaluate("smoke", str(kernel), self.workspace, self.op)
        if not smoke.passed:
            return smoke
        return evaluate("score", str(kernel), self.workspace, self.op)
