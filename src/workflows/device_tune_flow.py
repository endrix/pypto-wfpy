"""A device-guided edit, verify, profile and measure loop shared by operator workflows."""

from wfpy import File, connect, workflow

from src.agents.stage_author import stage_author
from src.lib.combo_config import CONNECTOR, IS_ACP, MODEL, TRANSPORT
from src.lib.optimize_config import TUNE_ROOT, _ensure_root
from src.tasks.device_tune_round import DeviceTuneRound
from src.viewers.markdown_viewer import MarkdownViewer
from src.viewers.swimlane_viewer import SwimlaneViewer


_PROMPT = """Study the fresh NPU trace, ranked tasks, kernel and history in the brief.
Use the assigned tuning skill to propose one change to the current PyPTO kernel.
Write the measured task or wait, its duration, a hypothesis and expected effect in evidence.
Keep the public API, benchmark inputs, precision, and instrumentation settings intact.
Work only by returning an edit on the Artifact port. The graph applies and measures it.
Return a JSON object with action "edit", evidence, old (one unique exact source fragment),
new (its replacement), and optional incore_task (one exact ranked task name for follow-up).
After swimlane tuning request in-core analysis only for a specific measured task.
If no evidence supports an edit, return {"action":"stop","evidence":"your measurement and reason"}."""


def make_device_tune_flow(workspace: str, adapter: str, budget: int = 3, device: int = 0):
    """Build the loop for a module implementing check(kernel, mode, device) and benchmark(kernel, device)."""
    if budget < 0 or device < 0:
        raise ValueError("device and edit budget must be nonnegative")
    if TUNE_ROOT.is_dir():
        _ensure_root(TUNE_ROOT.parent)
        _ensure_root(TUNE_ROOT)

    @workflow(inputs={"Kernel": File(ext=".py")},
              outputs={"Report": File(ext=".md"), "Winner": File(ext=".py"), "Status": str})
    def device_tuning():
        controller = DeviceTuneRound(workspace=workspace, adapter=adapter, budget=budget, device=device)
        options = dict(transport=TRANSPORT, connector=CONNECTOR if IS_ACP else "", model=MODEL)
        swim = stage_author("swimlane", _PROMPT, "tune-swimlane", ".json", **options)()
        incore = stage_author("incore", _PROMPT, "tune-incore", ".json", **options)()
        profile, diff, decision = MarkdownViewer(), MarkdownViewer(), MarkdownViewer()
        trace = SwimlaneViewer()
        connect("Kernel", controller.Kernel)
        connect(controller.Swim, swim.Brief)
        connect(swim.Artifact, controller.SwimEdit)
        connect(controller.Incore, incore.Brief)
        connect(incore.Artifact, controller.IncoreEdit)
        connect(controller.Profile, profile.In)
        connect(controller.Trace, trace.Swimlane)
        connect(controller.Diff, diff.In)
        connect(controller.Decision, decision.In)
        connect(controller.Winner, "Winner")
        connect(controller.Report, "Report")
        connect(controller.Status, "Status")

    return device_tuning
