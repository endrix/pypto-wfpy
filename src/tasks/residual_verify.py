"""Verify an authored implementation before admitting it into the device tuning loop."""

import ast
import json
from pathlib import Path

from src.lib.device_tune import measure
from src.lib.residual_contract import implementation_brief
from wfpy import File, Port, action, task


@task
class ResidualVerify:
    workspace: str
    device: int
    _attempt: int = 0

    class Ports:
        Artifact = Port[File](direction="in")
        Kernel = Port[File](direction="out", ext=".py")
        Retry = Port[File](direction="out", ext=".md")
        Failed = Port[File](direction="out", ext=".json")

    @action(consumes={"Artifact": 1}, produces={"Kernel": 1, "Retry": 1, "Failed": 1})
    def verify(self, artifact):
        self._attempt += 1
        workspace = Path(self.workspace)
        attempt = workspace / f"author-{self._attempt}"
        attempt.mkdir(parents=True, exist_ok=True)
        kernel = attempt / "kernel.py"
        try:
            kernel.write_bytes(Path(str(artifact)).read_bytes())
            source = kernel.read_text()
            tree = ast.parse(source, filename=str(kernel))
            decorators = (d for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                          for d in node.decorator_list if isinstance(d, ast.Call))
            if not any(ast.unparse(d.func) == "pypto.frontend.jit" for d in decorators):
                raise ValueError("expected a PyPTO Tensor mode @pypto.frontend.jit kernel")
            measure("verify", "src.lib.residual_eval", self.device, attempt / "npu", kernel)
        except (OSError, ValueError, SyntaxError) as exc:
            report = attempt / "failure.md"
            report.write_text(f"# Authoring verification failed\n\n{exc}\n\n"
                              f"Implementation: {kernel}\n\n{implementation_brief(workspace)}")
            if self._attempt == 1:
                return {"Retry": str(report)}
            failed = attempt / "failed.json"
            failed.write_text(json.dumps({"status": "FAILED", "kernel": str(kernel),
                                          "report": str(report)}))
            return {"Failed": str(failed)}
        return {"Kernel": str(kernel)}
