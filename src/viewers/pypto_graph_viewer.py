#!/usr/bin/env python3
# coding: utf-8
"""Node: `PyPTOGraphViewer` -- render one IR stage in the PyPTO Toolkit."""

from __future__ import annotations

from wfpy import File, Port, viewer


@viewer(
    action_name="command",
    inputs=["Graph"],
    command="PyPTO.Open.OpenRenderGraphForCurrent",
    args=["{uri}"],
)
class PyPTOGraphViewer:
    """Open one ``.json`` compute-graph dump in the PyPTO Toolkit.

    The Toolkit is driven by commands, not a disk-file custom editor: its
    ``PyPTO.open.editorView`` custom editor only claims the ``PyPTO:/`` URI
    scheme, so handing it a file path would not route there. The command
    ``PyPTO.Open.OpenRenderGraphForCurrent`` ("PyPTO Toolkit: Open file") is
    what renders a compute graph, which is why this is a ``command=`` viewer
    rather than a ``viewType=`` one.

    ``action_name="command"`` is what makes the IDE *run* that command: with the
    default ``"open"`` action the IDE would just open the file in the default
    editor -- which for a ``.json`` graph is plain text, not a render. ``args``
    hands the command the materialized graph file; ``{uri}`` is substituted with
    its ``file://`` URI.

    One instance per IR level lets a single run be read tensor -> tile -> block
    -> execute. To compare two stages instead of rendering one, the Toolkit also
    ships ``PyPTO.Open.OpenDiffComGraphForCurrent`` ("Open Compute Graph
    differences comparison").
    """

    class Ports:
        Graph = Port[File](direction="in")
