#!/usr/bin/env python3
# coding: utf-8
"""Node: `SwimlaneViewer` -- render the simulator's swimlane in the PyPTO Toolkit."""

from __future__ import annotations

from wfpy import File, Port, viewer


@viewer(
    action_name="command",
    inputs=["Swimlane"],
    command="PyPTO.Open.OpenSwimGraph",
    args=["{path}"],
)
class SwimlaneViewer:
    """Open ``merged_swimlane.json`` in the Toolkit's swimlane view.

    ``{path}`` and not ``{uri}``, deliberately. The command is registered as
    ``OpenSwimGraph(swimPath, seqNo, taskId)`` and does ``vscode.Uri.file(swimPath)``
    itself, so it wants a plain filesystem path; handing it a ``file://`` URI
    would produce a Uri for a path literally named "file:///...".

    That signature is also why this one works at all. Viewer command arguments
    cross the webview bridge as JSON, and the host revives a string into a
    ``vscode.Uri`` only for a fixed set of commands (`vscode.open`,
    `vscode.openWith`, `markdown.showPreview`, `markdown.showPreviewToSide`).
    Every other ``PyPTO.Open.*`` command takes a ``Uri`` object as its first
    argument and therefore cannot be driven this way -- ``OpenSwimGraph`` is the
    only one in the Toolkit that takes a path string.
    """

    class Ports:
        Swimlane = Port[File](direction="in")
