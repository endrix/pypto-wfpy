#!/usr/bin/env python3
# coding: utf-8
"""Node: `RouteAuthored` -- send a finished kernel on, or ask for a redesign."""

from __future__ import annotations

from pathlib import Path
import sys

from src.lib.author_config import MIN_ARTIFACT_BYTES
from wfpy import File, Port, action, guard, task


@task
class RouteAuthored:
    """Decide whether authoring produced something worth optimising.

    Two mutually exclusive guards derived from one predicate, so the exits cannot
    both pass: wfpy chooses nondeterministically among actions whose guards all
    hold, and declaration order is not a promise.

    An authoring run that produced no usable implementation is not a failure of
    the pipeline -- it is a result, and the loop should treat it the same way it
    treats a design whose tiling could not be saved: ask for a different one.
    """

    workspace: str
    op: str

    class Ports:
        Report = Port[File](direction="in")
        Kernel = Port[str](direction="out")
        Stalled = Port[File](direction="out", ext=".md")

    def _impl(self) -> Path:
        return Path(self.workspace) / f"{self.op}_impl.py"

    def _authored(self, report: str) -> bool:
        """Whether the workspace holds an implementation worth running."""
        impl = self._impl()
        return (impl.is_file() and impl.stat().st_size >= MIN_ARTIFACT_BYTES
                and any(line.startswith(f"**AUTHORED  {self.op}:")
                        for line in Path(str(report)).read_text().splitlines()[:3]))

    @action(consumes={"Report": 1}, produces={"Kernel": 1})
    @guard(lambda self, report: self._authored(report))
    def to_optimizer(self, report: str) -> str:
        impl = self._impl()
        print(f"[combo] authored {impl.name} ({impl.stat().st_size} bytes) -> optimising",
              file=sys.stderr)
        return str(impl)

    @action(consumes={"Report": 1}, produces={"Stalled": 1})
    @guard(lambda self, report: not self._authored(report))
    def stalled(self, report: str) -> str:
        print(f"[combo] no verified {self._impl().name}; forwarding the authoring failure",
              file=sys.stderr)
        return str(report)
