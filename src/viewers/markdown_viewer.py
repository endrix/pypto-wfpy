#!/usr/bin/env python3
# coding: utf-8
"""Node: `MarkdownViewer` -- open a report the workflow wrote."""

from __future__ import annotations

from wfpy import File, Port, viewer


@viewer(
    action_name="openWith",
    inputs=["Report"],
    viewType="vscode.markdown.preview.editor",
)
class MarkdownViewer:
    """Open a ``.md`` report in VS Code's markdown preview.

    ``openWith`` plus a ``viewType`` renders the file rather than showing its
    source, which is the point of asking the agents for markdown in the first
    place. The port name has to match the decorator's ``inputs``, or the viewer
    annotation names a port the class does not declare.

    Deliberately not tied to one report: the class carries no state and takes no
    parameters, so one instance per report is all it takes to show several.
    """

    class Ports:
        In = Port[File](direction="in")
