"""Frozen fixture execution, captured checkpoint comparison and scoped source edits."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import difflib
import json
import os
from pathlib import Path
import re
import subprocess

from src.lib import verify
from src.lib.runner import PYTHON, REPO_ROOT, execute_job, make_job
from src.lib.skill_paths import ops_root

REFERENCE = REPO_ROOT / "examples/05_attention_block.py"
SUSPECT = REPO_ROOT / "fixtures/05_attention_block_broken.py"
DRIVER = REPO_ROOT / "tools/check_precision_repair.py"


@dataclass(frozen=True)
class PrecisionCheck:
    kernel: str
    report: str
    passed: bool
    retryable: bool


def run_check(kernel: Path, directory: Path, seed: int | None = None):
    job = make_job(str(kernel), "npu", work_dir=directory)
    cmd = (PYTHON, "-u", str(DRIVER), str(kernel))
    if seed is not None:
        cmd += ("--seed", str(seed))
    result = execute_job(replace(job, cmd=cmd), int(os.getenv("PYPTO_PRECISION_REPAIR_TIMEOUT_S", "600")))
    (directory / "stdout.log").write_text(result.stdout)
    (directory / "stderr.log").write_text(result.stderr)
    return result


def _dump(kernel: Path, directory: Path, seed: int):
    directory.mkdir(parents=True, exist_ok=True)
    instrumented = directory / "instrumented.py"
    instrumented.write_text(verify.instrument(kernel.read_text()))
    result = run_check(instrumented, directory, seed)
    root = verify.find_verify_dir(directory)
    return result, (verify.read_dumps(root) if root else {})


def checkpoint_table(reference, suspect) -> tuple[list[str], str]:
    """Respect logical tensor axes when matching captures of lowered operations."""
    rows = ["## Aligned NPU checkpoints", "",
            f"Agreement threshold for matching tensors: {verify.AGREE_ATOL}.", "",
            "| Op magic | Reference op, shape and dtype | Suspect op, shape and dtype | Comparison |",
            "|---:|---|---|---|"]
    first = ""
    diffs = verify.compare(reference, suspect)
    culprit = verify.first_divergence(diffs)
    for row in diffs:
        magic = row.magic
        ref, sus = reference.get(magic), suspect.get(magic)
        lhs = f"{ref.opcode} {ref.shape} {ref.dtype}" if ref else "missing"
        rhs = f"{sus.opcode} {sus.shape} {sus.dtype}" if sus else "missing"
        comparison = row.note or f"max abs {row.max_abs:.6g}"
        if row is culprit:
            first = f"{magic} ({lhs} vs {rhs}, {comparison})"
        rows.append(f"| {magic} | {lhs} | {rhs} | {comparison} |")
    return rows, first


def evaluate(kernel: Path, directory: Path) -> PrecisionCheck:
    result = run_check(kernel, directory / "check")
    passed = result.passed and "PASS: all requested precision fixtures" in result.stdout.splitlines()
    lines = [f"# Device check for {kernel.name}", "", f"Candidate: `{kernel}`",
             f"Device: NPU {result.job.device_id}. Exit code {result.rc}.",
             "Fixed fp16 reference with absolute and relative tolerance 0.02.",
             "", "```text", result.stdout[-2600:] or result.error, "```"]
    match = re.search(r"^FAIL: precision fixture seed=(\d+)$", result.stdout, re.MULTILINE)
    retryable = False
    if result.rc == 2 and match:
        seed = int(match.group(1))
        golden, ref_dumps = _dump(REFERENCE, directory / "reference", seed)
        suspect, sus_dumps = _dump(kernel, directory / "suspect", seed)
        # Both instrumented runs must reproduce the original conditions.
        if golden.passed and suspect.rc == 2 and match.group() in suspect.stdout.splitlines() and ref_dumps and sus_dumps:
            table, first = checkpoint_table(ref_dumps, sus_dumps)
            lines += ["", *table]
            if first:
                lines += ["", f"First divergent checkpoint: {first}."]
            retryable = bool(first)
        else:
            lines += ["", "Checkpoint comparison unavailable.", f"Reference: {golden.error}",
                      f"Suspect: {suspect.error}", f"Dump counts: {len(ref_dumps)} vs {len(sus_dumps)}."]
    elif not passed:
        lines += ["", f"Device or compilation diagnostic: {result.error}", result.stderr[-1800:]]
    report = directory / "device-check.md"
    report.write_text("\n".join(lines) + "\n")
    return PrecisionCheck(str(kernel), str(report), passed, retryable)


def kernel_statements(source: str):
    """Expose editable statements in the JIT body, excluding fixture comments and the driver."""
    fn = next(n for n in ast.parse(source).body
              if isinstance(n, ast.FunctionDef) and n.name == "attention_block_kernel")
    return [node for node in fn.body if isinstance(node, (ast.Assign, ast.Expr))
            and not isinstance(getattr(node, "value", None), ast.Constant) and node.end_lineno == node.lineno]


def apply_edit(kernel: Path, artifact: Path, target: Path) -> str:
    source = kernel.read_text()
    edit = json.loads(artifact.read_text())
    if not isinstance(edit, dict) or not all(isinstance(edit.get(k), str) for k in ("old", "new")):
        raise ValueError('expected {"old": "current statement", "new": "replacement statement"}')
    old, new = edit["old"].strip(), edit["new"].strip()
    if not new or new == old or "\n" in new:
        raise ValueError("supply one changed statement")
    lines = source.splitlines(keepends=True)
    statements = kernel_statements(source)
    matching = [n for n in statements if lines[n.lineno-1].strip() == old]
    if len(matching) != 1:
        raise ValueError("old must uniquely match an executable line in attention_block_kernel")
    tree = ast.parse(new)
    node = matching[0]
    if len(tree.body) != 1 or type(tree.body[0]) is not type(node):
        raise ValueError("preserve the statement kind and supply one replacement")
    lines[node.lineno-1] = " " * node.col_offset + new + "\n"
    updated = "".join(lines)
    compile(updated, str(target), "exec")
    target.write_text(updated)
    return "".join(difflib.unified_diff(source.splitlines(True), updated.splitlines(True),
                                        fromfile=kernel.name, tofile=target.name))


def brief(check: PrecisionCheck) -> str:
    source = Path(check.kernel).read_text()
    lines = source.splitlines()
    statements = "\n".join(f"{s.lineno}: {lines[s.lineno-1].strip()}" for s in kernel_statements(source))
    review_script = ops_root() / "pypto-op-review/scripts/extract_pypto_calls.py"
    calls = ""
    if review_script.is_file():
        proc = subprocess.run([PYTHON, str(review_script), check.kernel], capture_output=True,
                              text=True, timeout=30)
        calls = proc.stdout if proc.returncode == 0 else proc.stderr
    # Let the diagnostician infer the boundary from the table. The deterministic
    # first difference remains in the device report for the reader to verify.
    measured = Path(check.report).read_text().partition("\nFirst divergent checkpoint:")[0]
    return (f"{measured}\n## Current JIT statements\n\n```python\n{statements}\n```\n"
            f"\n## Review skill call-site extraction\n\n{calls}")
