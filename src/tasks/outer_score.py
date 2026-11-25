#!/usr/bin/env python3
# coding: utf-8
"""Node: `OuterScore` -- the outer loop's only exit."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import sys

from src.lib import profile
from src.lib.jit_options import add_option
from src.lib.runner import REPO_ROOT
from wfpy import File, Port, action, guard, task


@task
class OuterScore:
    """Accept the result, or spend a redesign on it.

    One routing function, three exits, so they cannot overlap. A wfpy feedback
    edge has no backpressure and nothing ends it, so the counter here is what
    makes the outer loop terminate -- and it is charged whether the redesign
    came from a failed search or from authoring that produced nothing.

    `budget` is the number of REDESIGNS, not of rounds: 0 means author and
    optimise once, 1 means reconsider once if that found nothing. Each round is
    ten agents, so the cost is linear in this number.
    """

    budget: int
    workspace: str
    torch_source: str
    op: str

    _redesigns: int = 0

    class Ports:
        Searched = Port[File](direction="in")
        # The optimiser's own advisor already named what limits the kernel. It
        # arrives with the report -- two ports consumed together, ordinary
        # AND-firing for a task -- because it was previously produced, paid for
        # as an Opus call, and discarded: `combo_flow` did not connect it.
        Advice = Port[File](direction="in")
        Stalled = Port[File](direction="in")
        Evidence = Port[File](direction="out", ext=".md")
        Done = Port[File](direction="out", ext=".md")

    # ---- routing --------------------------------------------------------

    @staticmethod
    def _found_gain(report: str) -> bool:
        """Whether the optimisation search adopted anything.

        Read from the search's own report rather than recomputed: it already
        decided, against its own gain floor, and a second opinion here could
        disagree with the document the user reads.
        """
        try:
            text = Path(str(report)).read_text(encoding="utf-8")
        except OSError:
            return False
        return any(line.startswith("**OPTIMIZED  best=") for line in text.splitlines())

    def _redesign_left(self) -> bool:
        return self._redesigns < self.budget

    def _can_redesign_authoring(self, report: str) -> bool:
        # An exhausted reference check leaves no kernel design to reconsider.
        header = Path(str(report)).read_text(encoding="utf-8").splitlines()[:3]
        terminal = f"**INCOMPLETE  {self.op}: golden verification failed**"
        return self._redesign_left() and terminal not in header

    # ---- the evidence a redesign is based on ----------------------------

    def _evidence(self, report: str, why: str, advice: str = "") -> str:
        """The search report, the advisor's reading of it, the measured profile."""
        self._redesigns += 1
        parts = [
            f"# Redesign {self._redesigns} of {self.budget}", "",
            f"Operator: {self.op}", f"Original PyTorch source: {self.torch_source}",
            f"Workspace: {self.workspace}", f"Expected implementation: {self.op}_impl.py",
            "Preserve the source operator, signature, masks, and numerical contract on retry.",
            f"**Why:** {why}", "",
        ]
        if advice:
            try:
                parts += ["## What the optimiser's advisor concluded", "",
                          Path(str(advice)).read_text(encoding="utf-8")]
            except OSError:
                pass
        search_text = ""
        try:
            search_text = Path(str(report)).read_text(encoding="utf-8")
            parts += ["## Run report", "", search_text]
        except OSError:
            parts += ["## The optimisation search", "", "_(report unavailable)_"]

        # The measured device profile, if `./run.sh profile` has produced one.
        # Microseconds from hardware, unlike the search's cost-model cycles --
        # the prompt tells the agent not to compare them.
        impl = Path(self.workspace) / f"{self.op}_impl.py"
        winner = re.search(r"^\*\*(?:NO GAIN|OPTIMIZED)  best=.*? kernel=(.+?)\*\*$",
                           search_text, re.MULTILINE)
        if winner:
            candidate = Path(winner.group(1)).resolve()
            if candidate.is_relative_to(Path(self.workspace).resolve()):
                impl = candidate
        prof = REPO_ROOT / "runs" / "profile" / f"{impl.stem}_profile.md"
        measured = ""
        if prof.is_file() and impl.is_file():
            try:
                instrumented = add_option(impl.read_text(), "debug_options", profile.DEBUG_OPTIONS,
                                          note=profile.DEBUG_NOTE)
            except ValueError:
                instrumented = ""
            digest = sha256(instrumented.encode()).hexdigest()
            document = prof.read_text()
            if (instrumented and f"- instrumented SHA-256: `{digest}`" in document.splitlines()
                    and f"**PROFILED  {impl.stem}: passed**" in document.splitlines()):
                measured = document
        parts += ["", "## Measured device profile", "",
                  measured or f"No matching device profile for {self.op} and this implementation."]
        out = Path(self.workspace) / "redesign"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"evidence_{self._redesigns}.md"
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")
        return str(path)

    # ---- actions --------------------------------------------------------

    @action(consumes={"Searched": 1, "Advice": 1}, produces={"Done": 1})
    @guard(lambda self, report, advice: OuterScore._found_gain(report)
           or not self._redesign_left())
    def accept(self, report: str, advice: str) -> str:
        verdict = "a faster kernel" if self._found_gain(report) else "no gain, and no redesign left"
        print(f"[combo] done: {verdict}", file=sys.stderr)
        return str(report)

    @action(consumes={"Searched": 1, "Advice": 1}, produces={"Evidence": 1})
    @guard(lambda self, report, advice: not OuterScore._found_gain(report)
           and self._redesign_left())
    def reconsider(self, report: str, advice: str) -> str:
        print("[combo] the search finished without an accepted gain; reviewing the evidence",
              file=sys.stderr)
        return self._evidence(
            report, "the search finished without an accepted gain; see its diagnostics", advice)

    @action(consumes={"Stalled": 1}, produces={"Evidence": 1})
    @guard(lambda self, report: self._can_redesign_authoring(report))
    def from_stalled(self, report: str) -> str:
        return self._evidence(report, "authoring or verification failed; correct the reported failure")

    @action(consumes={"Stalled": 1}, produces={"Done": 1})
    @guard(lambda self, report: not self._can_redesign_authoring(report))
    def give_up(self, report: str) -> str:
        print("[combo] stopping after authoring failure; see the stage report", file=sys.stderr)
        return str(report)
