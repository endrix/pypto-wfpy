#!/usr/bin/env python3
# coding: utf-8
"""Structural summary of what the PyPTO compiler produced for a kernel.

A numeric self-check only proves the output values were right. It cannot tell
"compiled and computed correctly" apart from "compiled nothing at all" -- a file
whose kernel never ran still exits 0 -- and it says nothing about whether a fix
preserved the tiling. This module reads the compiler's own artifacts so a gate
can check both.

Dumping every pass costs ~230 files. Only four are needed to see all four IR
levels plus the final structure, which is what ``DUMP_PASSES`` selects.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

# One pass per IR level transition, plus the last pass in the pipeline.
# InferTensorFormat -> TENSOR_GRAPH, ExpandFunction -> TILE_GRAPH,
# SubgraphToFunction -> BLOCK_GRAPH + EXECUTE_GRAPH, CodegenPreproc -> final.
DUMP_PASSES = "InferTensorFormat,ExpandFunction,SubgraphToFunction,CodegenPreproc"

ALL_LEVELS = ("TENSOR_GRAPH", "TILE_GRAPH", "BLOCK_GRAPH", "EXECUTE_GRAPH")

# "Function k_PATH0_hiddenfunc_5[6] 1637... DYNAMIC_LOOP_PATH TILE_GRAPH {"
FUNC_RE = re.compile(r"^Function\s+(\S+?)(?:\[\d+\])?\s+.*\b(" + "|".join(ALL_LEVELS) + r")\b")
# "  RAWTENSOR[  3] <32 x 64 x DT_FP32> @18\"INCAST_LOCAL_BUF0\""
TILE_RE = re.compile(r"RAWTENSOR\[\s*\d+\]\s+<([0-9 x]+?)\s*x\s*DT_\w+>")


@dataclass(frozen=True)
class GraphSummary:
    """What the compiler made of one kernel."""

    compiled: bool
    levels: frozenset           # IR levels reached
    block_leaves: int           # distinct BLOCK_GRAPH functions == subgraph split
    tile_shapes: frozenset      # local-buffer tile shapes seen at TILE_GRAPH
    dump_root: str

    @property
    def levels_complete(self) -> bool:
        return set(ALL_LEVELS) <= set(self.levels)

    def problems(self) -> list[str]:
        """Unambiguous defects. Structural differences are NOT listed here."""
        issues = []
        if not self.compiled:
            issues.append("no kernel was compiled (program.json missing or trivial)")
        if not self.levels_complete:
            missing = [lv for lv in ALL_LEVELS if lv not in self.levels]
            issues.append(f"IR never reached {', '.join(missing)}")
        return issues

    def differences(self, other: GraphSummary) -> list[str]:
        """How this differs structurally from a baseline.

        Deliberately separate from ``problems()``: a change here is suspicious
        but can be entirely correct. Fixing a tiling bug *must* change the tile
        shapes, so treating any difference as a failure would reject the very
        repair that was asked for.
        """
        diffs = []
        if self.block_leaves != other.block_leaves:
            diffs.append(f"BLOCK_GRAPH leaves {other.block_leaves} -> {self.block_leaves}")
        if self.tile_shapes != other.tile_shapes:
            gone = sorted(other.tile_shapes - self.tile_shapes)
            new = sorted(self.tile_shapes - other.tile_shapes)
            parts = []
            if gone:
                parts.append(f"-{','.join(gone)}")
            if new:
                parts.append(f"+{','.join(new)}")
            diffs.append(f"tile shapes {' '.join(parts)}")
        return diffs


def _program_compiled(run_dir: Path) -> bool:
    """Whether program.json describes a real kernel rather than an empty shell."""
    for candidate in run_dir.glob("output/*/program.json"):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        # A kernel that never compiled leaves no functions/kernels entries.
        if isinstance(data, dict) and any(
            data.get(key) for key in ("functions", "kernels", "programs", "entries")
        ):
            return True
        if isinstance(data, dict) and len(data) > 2:
            return True
    return False


def summarize(run_dir: str | Path) -> GraphSummary:
    """Read a kernel's work dir and summarize the compiler's output."""
    root = Path(run_dir)
    compiled = _program_compiled(root)

    levels: set[str] = set()
    block_names: set[str] = set()
    tiles: set[str] = set()
    dump_root = ""

    dumps = sorted(root.glob("output/*/computation_graph/**/After_*.tifwkgr"))
    if dumps:
        dump_root = str(dumps[0].parent.parent)

    for dump in dumps:
        level = ""
        for line in dump.read_text(encoding="utf-8", errors="replace").splitlines():
            matched = FUNC_RE.match(line)
            if matched:
                name, level = matched.group(1), matched.group(2)
                levels.add(level)
                if level == "BLOCK_GRAPH":
                    block_names.add(name)
                continue
            if level == "TILE_GRAPH":
                tile = TILE_RE.search(line)
                if tile:
                    tiles.add(tile.group(1).replace(" ", ""))

    return GraphSummary(
        compiled=compiled,
        levels=frozenset(levels),
        block_leaves=len(block_names),
        tile_shapes=frozenset(tiles),
        dump_root=dump_root,
    )


# "Before_014_InsertOpForViewAssemble_k_PATH0_hiddenfunc_5.tifwkgr"
DUMP_RE = re.compile(r"^(Before|After)_(\d+)_(.+?)_(.*)\.tifwkgr$")


# The pass that dumps each IR level, keyed by the level name a viewer sees.
# This is the inverse of DUMP_PASSES: those four passes were chosen precisely
# because one lands on each level transition. The pass name is the reliable
# signal -- a late pass's dump can still carry an early level in its first
# `Function` header, so reading the header (as summarize does to *count*
# levels) would collapse several stages onto one file.
_LEVEL_PASS = {
    "TENSOR_GRAPH": "InferTensorFormat",
    "TILE_GRAPH": "ExpandFunction",
    "BLOCK_GRAPH": "SubgraphToFunction",
    "EXECUTE_GRAPH": "CodegenPreproc",
}


# "After_000_InferTensorFormat_row_softmax_kernel_PATH0_hiddenfunc_5.json"
# -> pass name "InferTensorFormat". The pass names of interest have no
# underscore, so the non-greedy capture stops at the one before the function.
_STAGE_DUMP_RE = re.compile(r"^After_\d+_(.+?)_")


def stage_dumps(run_dir: str | Path) -> dict[str, str]:
    """One representative ``.json`` graph dump per IR level, for a graph viewer.

    The compiler writes each pass graph as both a ``.tifwkgr`` text dump (what
    ``summarize`` parses to *count* levels) and a structured ``.json`` sibling.
    This returns the JSON to *open* for each level, so a run can be read tensor
    -> tile -> block -> execute. The level is taken from the dumping pass (see
    ``_LEVEL_PASS``), and within a pass the largest dump wins -- that is the
    whole-kernel graph rather than the tiny ``___entry___`` wrapper or a single
    per-leaf split emitted beside it. Keyed by level name; a level whose pass
    produced no dump is simply absent from the dict.
    """
    pass_level = {name: level for level, name in _LEVEL_PASS.items()}
    root = Path(run_dir)
    best: dict[str, tuple[int, Path]] = {}
    for dump in sorted(root.glob("output/*/computation_graph/**/After_*.json")):
        name_match = _STAGE_DUMP_RE.match(dump.name)
        level = pass_level.get(name_match.group(1)) if name_match else None
        if not level:
            continue
        size = dump.stat().st_size
        if level not in best or size > best[level][0]:
            best[level] = (size, dump)
    return {level: str(path) for level, (_, path) in best.items()}


def latest_graph_run(runs_root: str | Path) -> str:
    """The work dir of the most recent run under ``runs_root`` that dumped a graph.

    The repair loop gives each attempt its own work dir, and a GAVE UP run ends
    on an attempt that compiled nothing -- so that final dir has no dumps. This
    finds the newest attempt that DID produce a ``computation_graph`` (the last
    kernel that actually lowered), so the stage viewers show a real graph rather
    than falling back to the markdown report. Returns "" if none exist.
    """
    graphs = sorted(
        Path(runs_root).glob("*/output/*/computation_graph"),
        key=lambda p: p.stat().st_mtime,
    )
    # computation_graph -> output/<ts> -> output -> runs/<name> (the work dir).
    return str(graphs[-1].parents[2]) if graphs else ""


def level_of(path: Path) -> str:
    """Read the IR level out of a dump's `Function ...` header line."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("Function "):
                    found = re.search("|".join(ALL_LEVELS), line)
                    return found.group(0) if found else "?"
    except OSError:
        pass
    return "?"
