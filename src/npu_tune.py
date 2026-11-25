"""Profile and tune the verified KDA kernel using paired, synchronized NPU measurements."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfpy import File, connect, run, workflow

from src.lib.combo_config import CONNECTOR, IS_ACP
from src.lib.run_options import agent_run_options
from src.lib.runner import DEVICES, REPO_ROOT, WF_OUT
from src.sources.kernel_source import KernelSource
from src.viewers.markdown_viewer import MarkdownViewer
from src.workflows.device_tune_flow import make_device_tune_flow


@workflow(inputs={}, outputs={"Out": File(ext=".md"), "Status": str})
def npu_tune():
    kernel = KernelSource(path=File(str(REPO_ROOT / "examples/kda_decay_matrix_verified.py")))
    tune = make_device_tune_flow(str(REPO_ROOT / "custom" / "device_tune"), "src.lib.device_tune_kda",
                                 int(os.environ.get("PYPTO_DEVICE_TUNE_BUDGET", "3")), DEVICES[0])()
    report = MarkdownViewer()
    connect(kernel.Out, tune.Kernel)
    connect(tune.Report, report.In)
    connect(tune.Report, "Out")
    connect(tune.Status, "Status")


def main():
    result = run(npu_tune, out_dir=str(WF_OUT), acp_connector=CONNECTOR if IS_ACP else None,
                 **agent_run_options())
    for path in result["Out"]:
        print(Path(str(path)).read_text())
    return 0 if result["Status"] == ["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
