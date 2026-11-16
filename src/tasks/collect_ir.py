#!/usr/bin/env python3
# coding: utf-8
"""Pair Before_/After_ IR dumps and diff each pair."""

from __future__ import annotations

import difflib
from pathlib import Path

from src.graph import DUMP_RE, level_of
from src.records import IRRun, PassDiff
from wfpy import Port, action, task


@task
class CollectIR:
    """Group the per-pass dumps by (strategy, pass, function) and diff them."""

    class Ports:
        Run = Port[IRRun](direction="in")
        Diffs = Port[tuple](direction="out")

    @action(consumes={"Run": 1}, produces={"Diffs": 1})
    def collect(self, ir_run: IRRun) -> tuple:
        if not ir_run.dump_root:
            return (ir_run, ())

        root = Path(ir_run.dump_root)
        pairs: dict[tuple[str, int, str, str], dict[str, Path]] = {}
        for dump in root.rglob("*.tifwkgr"):
            matched = DUMP_RE.match(dump.name)
            if not matched:
                continue
            phase, index, pass_name, function = matched.groups()
            strategy = dump.relative_to(root).parts[0]
            pairs.setdefault((strategy, int(index), pass_name, function), {})[phase] = dump

        diffs: list[PassDiff] = []
        for (strategy, index, pass_name, function), sides in sorted(pairs.items()):
            before, after = sides.get("Before"), sides.get("After")
            if before is None or after is None:
                continue  # a pass that only dumped one side tells us nothing
            before_lines = before.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            after_lines = after.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            delta = list(difflib.unified_diff(
                before_lines, after_lines, fromfile=before.name, tofile=after.name, n=2))
            body = [ln for ln in delta
                    if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
            diffs.append(PassDiff(
                strategy=strategy,
                index=index,
                pass_name=pass_name,
                function=function,
                level_before=level_of(before),
                level_after=level_of(after),
                added=sum(1 for ln in body if ln.startswith("+")),
                removed=sum(1 for ln in body if ln.startswith("-")),
                diff_text="".join(delta),
            ))
        return (ir_run, tuple(diffs))
