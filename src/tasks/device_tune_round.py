"""Drive a bounded search from fresh device traces and record every challenger."""

import difflib
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import mkdtemp

from wfpy import File, Port, action, task

from src.lib.agent_io import read_json
from src.lib.device_tune import device_profile, measure, paired_verdict


_OUTPUTS = {"Swim": 1, "Incore": 1, "Report": 1, "Winner": 1, "Status": 1,
            "Profile": 1, "Trace": 1, "Diff": 1, "Decision": 1}


@task
class DeviceTuneRound:
    workspace: str
    adapter: str
    device: int
    budget: int

    _directory: str = ""
    _winner: str = ""
    _digest: str = ""
    _history: str = ""
    _round: int = 0
    _compared: int = 0
    _targets: tuple = ()

    class Ports:
        Kernel = Port[File](direction="in")
        SwimEdit = Port[File](direction="in")
        IncoreEdit = Port[File](direction="in")
        Swim = Port[File](direction="out", ext=".md")
        Incore = Port[File](direction="out", ext=".md")
        Report = Port[File](direction="out", ext=".md")
        Winner = Port[File](direction="out", ext=".py")
        Status = Port[str](direction="out")
        Profile = Port[File](direction="out", ext=".md")
        Trace = Port[File](direction="out", ext=".json")
        Diff = Port[File](direction="out", ext=".md")
        Decision = Port[File](direction="out", ext=".md")

    def _finish(self, status, reason):
        report = Path(self._directory) / "device-tuning.md"
        report.write_text(f"# NPU-guided tuning\n\n**{status}** {reason}\n\n"
                          f"Device: {self.device}\n\nWinner: `{self._winner}`\n\n"
                          f"SHA-256: `{self._digest}`\n\n{self._history}")
        return {"Report": str(report), "Status": status, "Winner": self._winner}

    def _profile(self, kernel, directory):
        return device_profile(self.adapter, self.device, directory, kernel)

    def _check(self, kernel, directory):
        for phase in ("smoke", "verify"):
            measure(phase, self.adapter, self.device, directory / phase, kernel)

    def _next(self, target="", *, stop=""):
        incumbent = Path(self._winner)
        if sha256(incumbent.read_bytes()).hexdigest() != self._digest:
            return self._finish("FAILED", "The selected kernel changed outside the measured edit.")
        if stop or (self._round and self._round >= self.budget):
            if not stop and not self._compared:
                return self._finish("FAILED", "No challenger completed evaluation before the budget expired.")
            return self._finish("PASS", stop or "Search budget reached.")
        directory = Path(self._directory) / f"round-{self._round + 1}"
        try:
            report, trace, names = self._profile(incumbent, directory / "incumbent-profile")
        except (OSError, ValueError, KeyError) as exc:
            return self._finish("FAILED", f"Fresh incumbent profile unavailable. {exc}")
        self._history += f"\nIncumbent profile: `{report}`\n"
        files = {"Trace": str(trace), "Profile": str(report)}
        if self.budget == 0:
            return {**files, **self._finish("PASS", "Verified baseline and captured the starting profile.")}
        self._round += 1
        self._targets = tuple(names)
        stage = "Incore" if target and target in names else "Swim"
        brief = directory / "brief.md"
        # Put the observations and source directly in the input. Agent sessions
        # can still open the full report for large profiles.
        detail = report.read_text()
        brief.write_text(f"# {stage} tuning round {self._round} of {self.budget}\n\n"
                         f"Device: {self.device}. Target task for in-core analysis: {target or 'none'}.\n"
                         f"Current longest measured tasks: {json.dumps(names)}\n"
                         f"Complete fresh profile: `{report}`\n\n{detail[:20000]}\n\n"
                         f"## Prior experiments\n{self._history}\n\n"
                         f"## Current kernel\n```python\n{incumbent.read_text()}\n```\n")
        return {**files, stage: str(brief)}

    @action(consumes={"Kernel": 1}, produces=_OUTPUTS)
    def initialize(self, kernel):
        root = Path(self.workspace)
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(mkdtemp(prefix="tune-", dir=root))
        self._directory = str(directory)
        baseline = directory / "baseline.py"
        self._winner = str(kernel)
        try:
            shutil.copyfile(str(kernel), baseline)
            self._winner = str(baseline)
            self._digest = sha256(baseline.read_bytes()).hexdigest()
            self._check(baseline, directory / "baseline-check")
        except (OSError, ValueError) as exc:
            return self._finish("FAILED", f"Baseline verification failed. {exc}")
        return self._next()

    def _apply(self, token, stage):
        directory = Path(self._directory) / f"round-{self._round}"
        candidate = directory / "challenger.py"
        incumbent = Path(self._winner)
        if sha256(incumbent.read_bytes()).hexdigest() != self._digest:
            return self._finish("FAILED", "The selected kernel changed during agent review.")
        files = {}
        target = stop = ""
        observations = []
        try:
            edit = read_json(token, required=("action", "evidence"))
            observations.append(str(edit["evidence"]))
            if edit["action"] == "stop":
                stop = f"{stage} agent stopped after reviewing this profile."
            elif edit["action"] == "edit":
                source = incumbent.read_text()
                old, new = edit.get("old"), edit.get("new")
                if not isinstance(old, str) or not isinstance(new, str) or not old or source.count(old) != 1:
                    raise ValueError("old must match exactly one nonempty source fragment")
                if old == new:
                    raise ValueError("edit leaves the kernel unchanged")
                changed = source.replace(old, new, 1)
                compile(changed, str(candidate), "exec", dont_inherit=True)
                candidate.write_text(changed)
                diff = directory / "diff.md"
                patch = difflib.unified_diff(source.splitlines(True), changed.splitlines(True),
                                             fromfile=incumbent.name, tofile=candidate.name)
                diff.write_text("# Proposed edit\n\n```diff\n" + "".join(patch) + "\n```\n")
                files["Diff"] = str(diff)
                self._check(candidate, directory / "candidate-check")
                timing = measure("paired", self.adapter, self.device, directory / "comparison",
                                 incumbent, candidate)
                samples = json.loads((timing / "pairs.json").read_text())
                adopt, reason = paired_verdict(samples)
                observations += [reason, f"Samples (ms), incumbent then challenger: {samples}"]
                report, trace, names = self._profile(candidate, directory / "challenger-profile")
                observations += [f"Challenger profile: `{report}`", f"Longest tasks: {names}"]
                files.update(Profile=str(report), Trace=str(trace))
                self._compared += 1
                if adopt:
                    self._winner = str(candidate)
                    self._digest = sha256(candidate.read_bytes()).hexdigest()
                observations.append("ADOPTED" if adopt else "KEPT INCUMBENT")
                requested = edit.get("incore_task", "")
                if requested in (names if adopt else self._targets):
                    target = requested
            else:
                raise ValueError("action must be edit or stop")
        except (OSError, SyntaxError, ValueError, TypeError, KeyError) as exc:
            observations.append(f"REJECTED: {exc}")
        decision = directory / "decision.md"
        decision.write_text(f"# Round {self._round} — {stage}\n\n" + "\n\n".join(observations) + "\n")
        self._history += f"\n{decision.read_text()}\nArtifacts: `{directory}`\n"
        files["Decision"] = str(decision)
        return {**files, **self._next(target, stop=stop)}

    @action(consumes={"SwimEdit": 1}, produces=_OUTPUTS)
    def apply_swim(self, edit):
        return self._apply(edit, "swimlane")

    @action(consumes={"IncoreEdit": 1}, produces=_OUTPUTS)
    def apply_incore(self, edit):
        return self._apply(edit, "in-core")
