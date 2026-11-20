#!/usr/bin/env python3
# coding: utf-8
"""Sub-workflow: show the three things the analysis produced.

    "PerfReport"   --> MarkdownViewer
    "BubbleReport" --> MarkdownViewer
    "Swimlane"     --> SwimlaneViewer

Sinks only, so the workflow declares no outputs. Viewers have no action of their
own and open a file on DISK, which is why every port here is a ``File`` and why
the analysis writes its reports out rather than passing strings along.

The two markdown reports share one viewer class, instantiated twice: the class
holds no state, so an instance is just "a place a report can be shown".

The swimlane gets its own class because it is not a file to open but a trace to
render, and that means asking the PyPTO Toolkit to do it -- see
``src/viewers/swimlane_viewer.py`` for why that one command can be driven from a
viewer node and the Toolkit's compute-graph commands cannot.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.viewers.markdown_viewer import MarkdownViewer  # noqa: E402
from src.viewers.swimlane_viewer import SwimlaneViewer  # noqa: E402
from wfpy import File, connect, workflow  # noqa: E402


@workflow(
    inputs={
        "PerfReport": File(ext=".md"),
        "BubbleReport": File(ext=".md"),
        "Swimlane": File(ext=".json"),
    },
    outputs={},
)
def visualize_flow() -> None:
    """Three sinks: two reports to read, one trace to render."""
    pipes = MarkdownViewer()
    cores = MarkdownViewer()
    swim = SwimlaneViewer()

    connect("PerfReport", pipes.In)
    connect("BubbleReport", cores.In)
    connect("Swimlane", swim.Swimlane)
