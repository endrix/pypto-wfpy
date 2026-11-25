"""Regressions for launcher settings and the CLI's terminal verdict."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from wfpy.core import WorkflowDef
from wfpy.runner import _build_workflow_graph

from src import kda_pipeline as launcher
from src.lib.records import SearchResult
from src.tasks.write_opt_report import WriteOptReport


def agent_specs(workflow):
    for actor in _build_workflow_graph(workflow).actors.values():
        if isinstance(actor.meta, WorkflowDef):
            yield from agent_specs(actor.meta)
        elif getattr(actor.meta, "agent_spec", None):
            yield actor.meta.agent_spec


class LauncherTest(unittest.TestCase):
    def test_settings_reach_every_agent(self):
        for connector, model in (("codex", "gpt-6.2-sol"), ("claude", "sonnet")):
            with self.subTest(connector=connector), patch.multiple(
                launcher, CONNECTOR=connector, MODEL=model, TRANSPORT="acp", IS_ACP=True
            ):
                agents = list(agent_specs(launcher.kda_pipeline._wfpy_workflow))
                self.assertEqual(len(agents), 10)
                self.assertEqual({(a.transport, a.connector, a.model) for a in agents},
                                 {("acp", connector, model)})
                self.assertTrue(all("in English" in a.prompt for a in agents))

    def test_only_completed_search_reports_succeed(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            writer = WriteOptReport(directory=directory)
            self.assertEqual(launcher.report_status([], require_npu=True), 1)
            npu = "Passed all five correctness fixtures on NPU 0.\n\nNPU latency: median 3.303 ms"
            for verdict, device_report, expected in (
                ("OPTIMIZED  best=100 cycles  kernel=k.py", npu, 0),
                ("NO GAIN  best=200 cycles  kernel=k.py", npu, 0),
                ("OPTIMIZED  best=100 cycles  kernel=k.py", "", 1),
                ("NPU FAILED  kernel=k.py", "fixture mismatch", 1),
                ("BASELINE UNUSABLE  no scored kernel", "", 1),
            ):
                with self.subTest(verdict=verdict, device_report=device_report):
                    result = SearchResult(verdict, directory, npu_report=device_report)
                    path = writer.write(result)["Report"]
                    self.assertEqual(launcher.report_status([path], require_npu=True), expected)
                    if verdict.startswith("OPTIMIZED"):
                        self.assertEqual(launcher.report_status([path], require_npu=False), 0)
            path = Path(directory) / "AUTHORING.md"
            path.write_text("# Authoring\n\n**INCOMPLETE**\n\n**OPTIMIZED  best=1**\n")
            self.assertEqual(launcher.report_status([path], require_npu=True), 1)
