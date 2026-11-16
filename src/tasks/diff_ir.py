#!/usr/bin/env python3
# coding: utf-8
"""Write the per-pass IR diffs to disk and render a summary report."""

from __future__ import annotations

from pathlib import Path

from wfpy import Port, action, task


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
