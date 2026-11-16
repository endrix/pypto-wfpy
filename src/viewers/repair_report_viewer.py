#!/usr/bin/env python3
# coding: utf-8
"""Node: `RepairReportViewer` -- opens the repair run's markdown report."""

from __future__ import annotations

from wfpy import File, Port, viewer


@viewer(action_name="open", inputs=["Report"], viewType="default")
class RepairReportViewer:
    """Open the report that WriteRepairReport left on disk.

    A viewer is a sink with no output and no action of its own: it opens a file
    on DISK, so the writer node has to run first. That is why this consumes the
    report's File token rather than the rendered string -- a token that is only
    a string in memory would leave this node with nothing to show.
    """

    class Ports:
        Report = Port[File](direction="in")
