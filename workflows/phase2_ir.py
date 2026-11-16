#!/usr/bin/env python3
# coding: utf-8
"""Phase 2 -- inspect how the PyPTO compiler rewrites the graph, pass by pass.

PyPTO lowers a kernel through four IR levels -- TENSOR_GRAPH -> TILE_GRAPH ->
BLOCK_GRAPH -> EXECUTE_GRAPH -- across ~39 passes, and with debug dumping on it
writes a readable `.tifwkgr` text dump of the graph both *before* and *after*
every pass. This workflow turns those dumps into a per-pass diff report, so you
can see exactly which pass changed what.

Four nodes, one per phase, mirroring phase 1:

    "Kernel" --> PrepareIR --> RunWithIR --> CollectIR --> DiffIR --> "Report"

- PrepareIR  builds the command, routed through tools/run_with_ir.py so the
             kernel source stays unmodified.
- RunWithIR  runs it and locates the dump root.
- CollectIR  pairs Before_*/After_* dumps per (strategy, pass, function) and
             reads each one's IR level out of its `Function ...` header.
- DiffIR     diffs each pair, writes the unified diffs to disk, and renders a
             summary: the IR level transitions plus the passes that changed the
             graph most.

Run (after `source /usr/local/Ascend/ascend-toolkit/set_env.sh`):

    python3 workflows/phase2_ir.py                              # default kernel
    python3 workflows/phase2_ir.py examples/03_fused_linear.py
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# wfpy loads this file by path; make sibling modules importable either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nodes import default_run_mode  # noqa: E402
from wfpy import Port, action, connect, run, task, workflow  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
IR_RUNNER = REPO_ROOT / "tools" / "run_with_ir.py"
DEFAULT_KERNEL = "examples/01_fused_elementwise.py"

# "Function k_PATH0_hiddenfunc_5[6] 1637... DYNAMIC_LOOP_PATH TILE_GRAPH {"
LEVEL_RE = re.compile(r"\b(TENSOR_GRAPH|TILE_GRAPH|BLOCK_GRAPH|EXECUTE_GRAPH)\b")
# "Before_014_InsertOpForViewAssemble_k_PATH0_hiddenfunc_5.tifwkgr"
DUMP_RE = re.compile(r"^(Before|After)_(\d+)_(.+?)_(.*)\.tifwkgr$")


@dataclass(frozen=True)
class IRJob:
    kernel: str
    name: str
    exists: bool
    work_dir: str
    cmd: tuple[str, ...]


@dataclass(frozen=True)
class IRRun:
    job: IRJob
    rc: int
    error: str
    seconds: float
    dump_root: str  # the computation_graph dir, or "" if absent


@dataclass(frozen=True)
class PassDiff:
    """One pass, for one function, before vs after."""

    strategy: str
    index: int
    pass_name: str
    function: str
    level_before: str
    level_after: str
    added: int
    removed: int
    diff_text: str

    @property
    def changed(self) -> bool:
        return self.added > 0 or self.removed > 0

    @property
    def level_shift(self) -> bool:
        return self.level_before != self.level_after


@dataclass(frozen=True)
class IRReport:
    job: IRJob
    rc: int
    error: str
    diffs: tuple[PassDiff, ...]
    report_path: str


def _level_of(path: Path) -> str:
    """Read the IR level out of a dump's `Function ...` header line."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("Function "):
                    found = LEVEL_RE.search(line)
                    return found.group(1) if found else "?"
    except OSError:
        pass
    return "?"


@task
class PrepareIR:
    """Build the IR-dumping command for one kernel."""

    run_mode: str

    class Ports:
        Kernel = Port[str](direction="in")
        Job = Port[IRJob](direction="out")

    @action(consumes={"Kernel": 1}, produces={"Job": 1})
    def prepare(self, kernel: str) -> IRJob:
        path = Path(kernel)
        if not path.is_absolute():
            path = REPO_ROOT / path
        name = path.stem
        work_dir = REPO_ROOT / "runs" / f"{name}_ir"
        work_dir.mkdir(parents=True, exist_ok=True)
        return IRJob(
            kernel=str(path),
            name=name,
            exists=path.is_file(),
            work_dir=str(work_dir),
            cmd=(sys.executable, str(IR_RUNNER), str(path), "--run_mode", self.run_mode),
        )


@task
class RunWithIR:
    """Run the kernel with dumping on, then locate the dump root."""

    timeout_s: int = 1800

    class Ports:
        Job = Port[IRJob](direction="in")
        Run = Port[IRRun](direction="out")

    @action(consumes={"Job": 1}, produces={"Run": 1})
    def execute(self, job: IRJob) -> IRRun:
        if not job.exists:
            return IRRun(job=job, rc=-1, error="kernel file not found", seconds=0.0, dump_root="")

        work = Path(job.work_dir)
        # Drop dumps from a previous run so the newest-dir pick below is honest.
        before = {p for p in work.glob("output/*") if p.is_dir()}

        started = time.time()
        try:
            proc = subprocess.run(
                list(job.cmd),
                cwd=job.work_dir,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            rc, stdout, stderr = -2, "", f"timeout after {self.timeout_s}s"

        error = ""
        if rc != 0:
            fail = [ln for ln in stdout.splitlines() if ln.startswith("FAIL:")]
            error = fail[0] if fail else (stderr.strip().splitlines() or ["no diagnostic output"])[-1]

        # The run stamps a fresh output/<timestamp>/ dir; take the new one.
        fresh = [p for p in work.glob("output/*") if p.is_dir() and p not in before]
        candidates = fresh or [p for p in work.glob("output/*") if p.is_dir()]
        dump_root = ""
        for cand in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
            if (cand / "computation_graph").is_dir():
                dump_root = str(cand / "computation_graph")
                break

        return IRRun(job=job, rc=rc, error=error, seconds=round(time.time() - started, 1), dump_root=dump_root)


@task
class CollectIR:
    """Pair Before_/After_ dumps and diff each pair."""

    class Ports:
        Run = Port[IRRun](direction="in")
        Diffs = Port[tuple](direction="out")

    @action(consumes={"Run": 1}, produces={"Diffs": 1})
    def collect(self, ir_run: IRRun) -> tuple:
        if not ir_run.dump_root:
            return (ir_run, ())

        root = Path(ir_run.dump_root)
        # (strategy, index, pass_name, function) -> {"Before": path, "After": path}
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
            delta = list(
                difflib.unified_diff(
                    before_lines, after_lines, fromfile=before.name, tofile=after.name, n=2
                )
            )
            body = [ln for ln in delta if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
            diffs.append(
                PassDiff(
                    strategy=strategy,
                    index=index,
                    pass_name=pass_name,
                    function=function,
                    level_before=_level_of(before),
                    level_after=_level_of(after),
                    added=sum(1 for ln in body if ln.startswith("+")),
                    removed=sum(1 for ln in body if ln.startswith("-")),
                    diff_text="".join(delta),
                )
            )
        return (ir_run, tuple(diffs))


@task
class DiffIR:
    """Write the diffs to disk and render a summary."""

    top_n: int = 8

    class Ports:
        Diffs = Port[tuple](direction="in")
        Report = Port[str](direction="out")

    @action(consumes={"Diffs": 1}, produces={"Report": 1})
    def render(self, payload: tuple) -> str:
        ir_run, diffs = payload
        job = ir_run.job

        if ir_run.rc != 0:
            return f"FAIL  {job.name}: rc={ir_run.rc}  {ir_run.error[:80]}"
        if not diffs:
            return f"FAIL  {job.name}: ran clean but produced no IR dumps (is debug dumping on?)"

        out_dir = Path(job.work_dir) / "ir_diffs"
        out_dir.mkdir(parents=True, exist_ok=True)
        changed = [d for d in diffs if d.changed]

        lines: list[str] = [
            f"# IR pass report -- {job.name}",
            "",
            f"- kernel: `{job.kernel}`",
            f"- dump root: `{ir_run.dump_root}`",
            f"- pass/function dumps compared: {len(diffs)}  (changed: {len(changed)})",
            f"- wall time: {ir_run.seconds}s",
            "",
            "## IR level transitions",
            "",
        ]

        shifts = [d for d in diffs if d.level_shift]
        if shifts:
            lines.append("| strategy | pass | function | from | to |")
            lines.append("|---|---|---|---|---|")
            for d in shifts:
                lines.append(
                    f"| {d.strategy} | {d.index:02d} {d.pass_name} | `{d.function}` "
                    f"| {d.level_before} | {d.level_after} |"
                )
        else:
            lines.append("_No pass changed the IR level for any function._")

        # A level belongs to a (pass, function) pair, not to a pass: after
        # SubgraphToFunction several functions coexist at different levels, so
        # grouping by pass alone would list one pass under three levels at once.
        per_level: dict[str, dict[str, list[int]]] = {}
        for d in diffs:
            per_level.setdefault(d.level_after, {}).setdefault(d.function, []).append(d.index)
        lines += ["", "## IR level reached, per function", ""]
        lines.append("| level | function | passes at this level |")
        lines.append("|---|---|---|")
        for level in ("TENSOR_GRAPH", "TILE_GRAPH", "BLOCK_GRAPH", "EXECUTE_GRAPH", "?"):
            for function, indices in sorted(per_level.get(level, {}).items()):
                span = f"{min(indices):02d}-{max(indices):02d}" if len(indices) > 1 else f"{indices[0]:02d}"
                lines.append(f"| {level} | `{function}` | {len(indices)} (pass {span}) |")

        lines += ["", f"## Largest graph changes (top {self.top_n})", "",
                  "| pass | function | level | +lines | -lines | diff |", "|---|---|---|---|---|---|"]
        for d in sorted(changed, key=lambda x: x.added + x.removed, reverse=True)[: self.top_n]:
            fname = f"{d.strategy}_{d.index:02d}_{d.pass_name}_{d.function}.diff"
            (out_dir / fname).write_text(d.diff_text, encoding="utf-8")
            lines.append(
                f"| {d.index:02d} {d.pass_name} | `{d.function}` | {d.level_after} "
                f"| +{d.added} | -{d.removed} | `ir_diffs/{fname}` |"
            )

        # Keep every changed pass on disk, not just the top N shown above.
        for d in changed:
            fname = f"{d.strategy}_{d.index:02d}_{d.pass_name}_{d.function}.diff"
            path = out_dir / fname
            if not path.exists():
                path.write_text(d.diff_text, encoding="utf-8")

        report_path = Path(job.work_dir) / "ir_report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        shift_note = ", ".join(f"{d.level_before}->{d.level_after}@{d.pass_name}" for d in shifts) or "none"
        return (
            f"PASS  {job.name}: {len(changed)}/{len(diffs)} passes changed the graph; "
            f"level shifts: {shift_note}\n"
            f"      report: {report_path}\n"
            f"      diffs:  {out_dir}"
        )


@workflow(inputs={"Kernel": str}, outputs={"Report": str})
def inspect_ir() -> None:
    """Wire the four IR phases together."""
    prepare = PrepareIR(run_mode=default_run_mode())
    runner = RunWithIR()
    collect = CollectIR()
    differ = DiffIR()

    connect("Kernel", prepare.Kernel)
    connect(prepare.Job, runner.Job)
    connect(runner.Run, collect.Run)
    connect(collect.Diffs, differ.Diffs)
    connect(differ.Report, "Report")


def main() -> int:
    kernels = sys.argv[1:] or [DEFAULT_KERNEL]
    failures = 0
    for kernel in kernels:
        result = run(inspect_ir, inputs={"Kernel": kernel})
        for line in result["Report"] or ["(no output)"]:
            print(line)
            if line.startswith(("FAIL", "(no")):
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
