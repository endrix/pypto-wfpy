#!/usr/bin/env python3
# coding: utf-8
"""Node: `MarkdownViewer` -- open a report the workflow wrote."""

from __future__ import annotations

from wfpy import File, Port, viewer


@viewer(action_name="open", inputs=["Report"], viewType="default")
class MarkdownViewer:
    """Open a ``.md`` report in the editor.

    Deliberately not tied to one report: the class carries no state and takes no
    parameters, so one instance per report is all it takes to show several.
    Each instance opens whatever lands on its own ``Report`` port.
    """

    class Ports:
        Report = Port[File](direction="in")
