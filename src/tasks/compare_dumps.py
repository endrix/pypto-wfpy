#!/usr/bin/env python3
# coding: utf-8
"""Node: `CompareDumps` -- find the first op where two runs disagree."""

from __future__ import annotations

import sys

from src.lib import verify
from src.lib.records import Divergence, DumpSet
from wfpy import Port, action, task


@task
class CompareDumps:
    """Align two runs' op dumps and name the earliest divergence.

    Two input ports consumed together -- ordinary AND-firing for a `@task`,
    and correct here because one reference run and one suspect run arrive per
    triage and the comparison needs both.

    Deterministic on purpose. No agent takes part in the localisation: the ops
    are aligned by op magic, which is stable across runs, and compared
    elementwise. An agent is asked afterwards to explain the op this names,
    which is judgement; choosing it is arithmetic.
    """

    class Ports:
        Reference = Port[DumpSet](direction="in")
        Suspect = Port[DumpSet](direction="in")
        Divergence = Port[Divergence](direction="out")

    @action(consumes={"Reference": 1, "Suspect": 1}, produces={"Divergence": 1})
    def compare(self, reference: DumpSet, suspect: DumpSet) -> Divergence:
        names = (reference.job.kernel, suspect.job.kernel)

        missing = [
            f"{label} produced no op dumps under {ds.verify_dir or '(no verify dir)'}"
            for label, ds in (("reference", reference), ("suspect", suspect))
            if not ds.found
        ]
        if missing:
            # Said rather than raised: no dumps usually means the run never
            # reached the device, and the reason is worth carrying to the report.
            print(f"[triage] {'; '.join(missing)}", file=sys.stderr)
            return Divergence(reference=names[0], suspect=names[1], diffs=(),
                              culprit=None, error="; ".join(missing))

        if reference.passed and not suspect.passed:
            pass  # the expected shape: a good reference and a failing suspect
        elif not reference.passed:
            print("[triage] warning: the reference failed its own check too, so "
                  "'first divergence' means 'first difference', not 'first fault'",
                  file=sys.stderr)

        diffs = verify.compare(reference.dumps, suspect.dumps)
        culprit = verify.first_divergence(diffs)
        where = f"op {culprit.magic} {culprit.opcode}" if culprit else "nowhere"
        print(
            f"[triage] {len(diffs)} ops aligned; first divergence at {where}",
            file=sys.stderr,
        )
        return Divergence(reference=names[0], suspect=names[1],
                          diffs=tuple(diffs), culprit=culprit)
