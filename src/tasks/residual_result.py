"""Produce one branch result after authoring failure or completion of tuning."""

import json
from pathlib import Path
from wfpy import File, Port, action, task


@task
class ResidualResult:
    workspace: str

    class Ports:
        Kernel = Port[File](direction="in")
        Report = Port[File](direction="in")
        Status = Port[str](direction="in")
        Failed = Port[File](direction="in")
        PlanFailed = Port[File](direction="in")
        DesignFailed = Port[File](direction="in")
        ReviewFailed = Port[File](direction="in")
        Result = Port[File](direction="out", ext=".json")

    @action(consumes={"Kernel": 1, "Report": 1, "Status": 1}, produces={"Result": 1})
    def finished(self, kernel, report, status):
        path = Path(self.workspace) / "result.json"
        path.write_text(json.dumps({"status": status, "kernel": str(kernel), "report": str(report)}))
        return str(path)

    @action(consumes={"Failed": 1}, produces={"Result": 1})
    def failed(self, error):
        return str(error)


    @action(consumes={"PlanFailed": 1}, produces={"Result": 1})
    def plan_failed(self, error):
        return str(error)

    @action(consumes={"DesignFailed": 1}, produces={"Result": 1})
    def design_failed(self, error):
        return str(error)

    @action(consumes={"ReviewFailed": 1}, produces={"Result": 1})
    def review_failed(self, error):
        return str(error)
