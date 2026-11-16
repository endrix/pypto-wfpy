#!/usr/bin/env python3
# coding: utf-8
"""Parsing for the simulator's performance report. Plain helpers, no actors."""

from __future__ import annotations

import json
from pathlib import Path
import re

# "Average MTE_IN Usage............................:   4387.00   19.86%"
# "Total Cycles....................................:     22087"
STAT_RE = re.compile(r"^\s*\|?-*\s*(.+?)\.{2,}:\s*([\d.]+)(?:\s+([\d.]+)%)?\s*$")

# The pipes a tile kernel can be limited by, in report order.
PIPES = ("Cube", "Vector", "MTE_IN", "MTE1", "MTE_OUT")

# Counters worth surfacing out of the many the report prints.
INTERESTING = (
    "Total Submit Tasks", "Cube Tasks", "Vector Tasks",
    "Average Task Cycles", "Max Task Cycles", "Min Task Cycles",
    "Core Machine Count", "Cube Count", "Vector Count",
)


def parse_stat_report(path: Path) -> tuple[float, dict, dict, dict]:
    """Parse the MachineModel text report into (cycles, pipe%, pipe_cycles, counters)."""
    total_cycles = 0.0
    pipe_pct: dict[str, float] = {}
    pipe_cycles: dict[str, float] = {}
    counters: dict[str, float] = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        matched = STAT_RE.match(line)
        if not matched:
            continue
        label, value, pct = matched.group(1).strip(), float(matched.group(2)), matched.group(3)

        # "Total Cycles" appears once per section; the first is the whole run.
        if label == "Total Cycles" and not total_cycles:
            total_cycles = value

        pipe = next((p for p in PIPES if label == f"Average {p} Usage"), None)
        if pipe is not None:
            pipe_cycles[pipe] = value
            if pct is not None:
                pipe_pct[pipe] = float(pct)
        elif pct is None:
            counters.setdefault(label, value)

    return total_cycles, pipe_pct, pipe_cycles, counters


def parse_tile_ops(path: Path) -> dict:
    """Pull per-tile-op work amounts out of the perf metrics JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    paths = (data.get("summary") or {}).get("tileOpLongestPaths") or {}
    return {name: info.get("amountLp", 0) for name, info in paths.items() if isinstance(info, dict)}
