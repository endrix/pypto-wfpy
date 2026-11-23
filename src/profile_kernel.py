#!/usr/bin/env python3
# coding: utf-8
"""Profile a PyPTO kernel on the NPU and report what the device actually did.

Three nodes, one of them a sub-workflow:

    KernelSource --> collect_flow --> WriteProfileReport --+--> MarkdownViewer
                                                           |
                                                           +--> "Out"

This is cannbot's S2_COLLECT and S-1 (core usage analysis) as a workflow. The
kernel is run with `debug_options={"runtime_debug_mode": 1}`, which makes PyPTO
write `merged_swimlane.json`, `program.json`, `bubble_analysis.log` and
`tilefwk_L1_prof_data.json`; the tuning skill's own `analyze_swimlane.py` and
`analyze_core_usage.py` then read them.

**These are measured microseconds.** Every other performance number in this
project -- `perf`, `sim-analyze`, the objective `optimize` minimises -- is cycles
from the Soc CAModel, a cost model running on the CPU. This workflow is the only
one that reports what the hardware did, and the two are not comparable.

On `05_attention_block` it says AIC 4/24 cores at 17% and AIV 2/48 at 4%, with
`FILL CORES FIRST (reduce TileShape)` per subgraph -- confirming on hardware the
under-occupancy the cost model had already suggested, and arriving at the
catalogue's own prescribed order: S-1 before S-2.

NPU only, and not configurable: the data is copied off the device, so there is
nothing to copy in sim.

Run:
    ./run.sh profile                                  # 05_attention_block
    ./run.sh profile examples/03_fused_linear.py
    PYPTO_KERNEL=examples/04_layer_norm.py ./run.sh profile
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lib import cann  # noqa: E402
from src.lib.runner import REPO_ROOT, WF_OUT  # noqa: E402
from src.sources.kernel_source import KernelSource  # noqa: E402
from src.tasks.run_profile import RUN_MODE  # noqa: E402
from src.tasks.write_profile_report import WriteProfileReport  # noqa: E402
from src.viewers.markdown_viewer import MarkdownViewer  # noqa: E402
from src.workflows.collect_flow import collect_flow  # noqa: E402
from wfpy import File, connect, run, workflow  # noqa: E402

DEFAULT_KERNEL = "examples/05_attention_block.py"


def _requested_kernel() -> str:
    """Which kernel to profile.

    PYPTO_KERNEL, else argv[1] when this file is the program. Not argv alone:
    the source's path is a constructor parameter, so the graph needs it at BUILD
    time, and whoever imports this module owns argv.
    """
    from_env = os.environ.get("PYPTO_KERNEL", "").strip()
    if from_env:
        return from_env
    if __name__ == "__main__" and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]
    return DEFAULT_KERNEL


_requested = _requested_kernel()
KERNEL_PATH = _requested if Path(_requested).is_absolute() else str(REPO_ROOT / _requested)


@workflow(inputs={}, outputs={"Out": str})
def profile_kernel() -> None:
    """Three nodes: the kernel, the device run, and somewhere to read it."""
    kernel = KernelSource(path=File(KERNEL_PATH))
    collect = collect_flow("profile", slot=0)()
    writer = WriteProfileReport()
    view = MarkdownViewer()

    connect(kernel.Out, collect.Kernel)
    connect(collect.Profile, writer.Profile)
    connect(writer.Report, view.In)
    connect(writer.Verdict, "Out")


def main() -> int:
    # Makes an IDE run behave like a run.sh run: run.sh sources set_env.sh, a
    # direct launch from an IDE does not.
    cann.apply_to_process()
    print(f"[profile] kernel={KERNEL_PATH}\n[profile] mode={RUN_MODE} (pinned)",
          file=sys.stderr)
    result = run(profile_kernel, out_dir=str(WF_OUT))
    lines = result["Out"] or ["(no output)"]
    for line in lines:
        print(line)
    return 1 if any(ln.startswith("NO DATA") for ln in lines) else 0


if __name__ == "__main__":
    sys.exit(main())
