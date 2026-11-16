#!/usr/bin/env python3
# coding: utf-8
"""Node: `KernelSource` -- the @source at the head of a repair run."""

from __future__ import annotations

from wfpy import File, Port, source


@source(viewType="default")
class KernelSource:
    """The PyPTO kernel under repair, emitted once at the head of the run.

    A source has no inputs, so nothing can trigger it and nothing can stop it;
    wfpy writes the emit-once latch for us.

    ``path`` is a PARAMETER, not state: an annotated field *with* a class-level
    default would become state shared by every instance, so two runs in one
    process would read the same file. At runtime a File token is simply the
    resolved path string, which is why this can feed a Port[str] input.
    """

    path: File

    class Ports:
        Out = Port[File](direction="out")
