#!/usr/bin/env python3
# coding: utf-8
"""Node: `MergeBrief` -- the head of the outer loop, where two edges become one."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
import sys

from wfpy import File, Port, action, task


@task
class MergeBrief:
    """Forward whichever brief arrived: the first one, or a redesign.

    Here for the same reason `MergeKernel` is: a wfpy input port does not fan in,
    and the authoring sub-workflow's boundary has one `Source` port, so the
    initial PyTorch module and the loop's feedback edge cannot both connect to
    it. Putting the join in its own node keeps `author_flow` reusable and makes
    the outer loop visible in the graph rather than hidden inside a task.
    """

    op: str
    workspace: str
    torch_source: str
    artifact: str
    _round: int = 0

    class Ports:
        Initial = Port[File](direction="in")
        Retry = Port[File](direction="in")
        # .md: the brief forwarded here is either the PyTorch source or a
        # redesign brief, and the materialized copy is named from this.
        Out = Port[File](direction="out", ext=".md")

    def _forward(self, brief: str, how: str) -> str:
        self._round += 1
        print(f"[combo] round {self._round}: authoring from the {how}", file=sys.stderr)
        return str(brief)

    @action(consumes={"Initial": 1}, produces={"Out": 1})
    def from_initial(self, brief: str) -> str:
        return self._forward(brief, "initial brief")

    @action(consumes={"Retry": 1}, produces={"Out": 1})
    def from_retry(self, brief: str) -> str:
        directory = Path(self.workspace) / "retries"
        directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", dir=directory, prefix="brief-", suffix=".md",
                                encoding="utf-8", delete=False) as out:
            out.write(f"Operator: {self.op}\nOriginal source: {self.torch_source}\n"
                      f"Workspace: {self.workspace}\nArtifact to write: {self.artifact}\n\n")
            out.write(Path(str(brief)).read_text(encoding="utf-8"))
        return self._forward(out.name, "retry brief")
