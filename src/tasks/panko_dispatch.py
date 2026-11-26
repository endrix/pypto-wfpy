"""Dispatch the vendor's CODE/STOP protocol and verify its delivered artifact."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

from src.lib.device_tune import compare_kda_pair, measure
from src.lib import panko
from src.lib.agent_io import read_json
from src.lib.skill_paths import ops_root
from wfpy import File, Port, action, task


@task
class PankoDispatch:
    workspace: str
    device: int
    budget: int
    tile_search: bool
    _calls: int = 0

    class Ports:
        Seed = Port[File](direction="in")
        Directive = Port[File](direction="in")
        Source = Port[File](direction="in")
        Tune = Port[str](direction="out")
        Code = Port[str](direction="out")
        Report = Port[File](direction="out", ext=".md")
        Frontier = Port[File](direction="out", ext=".md")

    def _report(self, success: bool, message: str) -> dict:
        root = Path(self.workspace)
        try:
            if (root / "dispatch.json").is_file() and not success:
                try:
                    panko.close_workers(root)
                except (OSError, subprocess.SubprocessError) as exc:
                    message += f"\n\nFailed to close the campaign workers: {exc}"
            with NamedTemporaryFile("w", dir=root if (root / "dispatch.json").is_file() else None,
                                    prefix="panko-report-", suffix=".md", delete=False) as report:
                report.write(f"# PANKO search\n\n**{'VERIFIED' if success else 'INCOMPLETE'}**\n\n"
                             f"{message}\n\nWorkspace: `{root}`\n")
                return {"Report": report.name}
        finally:
            self._lock.close()

    def _frontier(self, tree, directive) -> str:
        directory = Path(self.workspace) / "decisions"
        directory.mkdir(exist_ok=True)
        with NamedTemporaryFile("w", dir=directory, suffix=".md", delete=False) as report:
            progress = tree["progress"]
            report.write(f"# PANKO decision\n\n{directive['action_id']}: {directive['intent']}\n\n"
                         f"Best node: {tree['best_node']}. Evaluations charged: {progress['evals_used']}.\n\n"
                         "## Open intents\n\n")
            for node in tree["nodes"]["open"]:
                report.write(f"- {node['id']} ({node.get('stage', '')}): {node.get('delta', '')}\n")
            return report.name

    def _finish(self, state: dict, reason: str | None = None) -> dict:
        root = Path(self.workspace)
        progress = panko.terminal(root, state)
        if reason is not None and progress["stop_reason"] != reason:
            raise ValueError("STOP directive disagrees with the persisted harness reason")
        try:
            panko.close_workers(root)
        except subprocess.SubprocessError as exc:
            raise ValueError(f"Could not release the resident workers for final NPU verification: {exc}") from exc
        logs = panko.verify_winner(root, state)
        timing = next((line for line in (Path(logs) / "stdout.log").read_text().splitlines()
                       if line.startswith("NPU latency: median ")), None)
        if timing is None:
            raise ValueError(f"Missing final timing in {logs}")
        selected = root / "winner.py"
        comparison = ""
        pending = panko.pending_refinement(root, progress)
        if pending is not None:
            try:
                pending_logs = Path(panko.verify_winner(root, state, candidate=pending))
            except panko.FinalCheckFailed as exc:
                comparison = f"\n\nPending refinement `{pending}` was ineligible. {exc}"
            else:
                challenger = pending_logs / "winner.py"
                adopt, evidence = compare_kda_pair(selected, challenger, pending_logs / "paired", self.device)
                if adopt:
                    selected = challenger
                comparison = (f"\n\nPending refinement: `{pending}` "
                              f"({progress['refine']['best']['p']} µs profiled). Passed all five NPU fixtures. "
                              f"Paired timings (harness winner → pending): {evidence}")
        comparison += f"\n\nRecommended deployment kernel: `{selected}`."
        rival = os.environ.get("PYPTO_PANKO_COMPARE_KERNEL")
        if rival:
            adapter = "src.lib.device_tune_kda"
            device = state["manifest"]["device"]
            candidate = Path(rival).resolve()
            directory = Path(logs) / "comparison"
            measure("verify", adapter, device, directory / "rival-check", candidate)
            faster, evidence = compare_kda_pair(candidate, selected, directory / "paired", device)
            comparison += (f"\n\nPaired comparison with `{candidate}`: {evidence} "
                           f"Selected: `{selected if faster else candidate}`.")
        return self._report(True, f"Stop reason: {progress['stop_reason']}\n\n"
                            f"NPU AICore latency: {progress['preopt_us']} → "
                            f"{progress['best_latency_us']} µs. Evaluations charged: {progress['evals_used']}.\n\n"
                            f"Measured source: `{panko.kernel(root)}`\n\nDelivered source: `{root / 'winner.py'}`\n\n"
                            f"Five-fixture NPU verification: `{logs}`\n\n{timing} (profiling off)\n\n"
                            f"Search tree: `{root / 'optimization/search_state.json'}`{comparison}")

    def _next(self, state: dict) -> dict:
        root = Path(self.workspace)
        if (root / "optimization/controller-terminal.json").is_file():
            return self._finish(state)
        if state["phase"] == "publish":
            # By checkpointing the reply before replacing the file, replaying a crash
            # between these two writes is idempotent.
            panko.write(panko.kernel(root), (root / "pending.py").read_bytes())
            state["phase"] = "step"
            panko.checkpoint(root, state)
        self._calls += 1
        if self._calls > 12 * self.budget + 24:
            return self._report(False, "Dispatch limit reached. Resume this workspace to continue the saved search.")
        if state["phase"] == "code":
            return {"Code": f"Action {state['action_id']}: {state['intent']}\n\n"
                            + panko.kernel(root).read_text()}
        return {"Tune": panko.request(root, state)}

    @action(consumes={"Seed": 1}, produces={"Tune": 1, "Code": 1, "Report": 1})
    def start(self, seed: str) -> dict:
        root = Path(self.workspace)
        root.mkdir(parents=True, exist_ok=True)
        self._lock = (root / ".wfpy.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError(f"This PANKO workspace already has an active runner: {root}") from None
        try:
            if self.budget < 1:
                raise ValueError("PYPTO_PANKO_BUDGET must be positive")
            if not (ops_root() / panko.MANUAL / "scripts/harness_session.py").is_file():
                raise ValueError("Set CANNBOT_OPS to a collection containing the PANKO skill and harness")
            state = panko.setup(root, Path(str(seed)), self.device, self.budget, self.tile_search)
            if self.tile_search and importlib.util.find_spec("optuna") is None:
                raise ValueError("PANKO Bayesian tile search requires optuna in the workflow venv")
            return self._next(state)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            return self._report(False, str(exc))

    @action(consumes={"Directive": 1}, produces={"Tune": 1, "Code": 1, "Report": 1, "Frontier": 1})
    def route(self, token: str) -> dict:
        root = Path(self.workspace)
        try:
            state = json.loads((root / "dispatch.json").read_text())
            directive = read_json(token, required=("directive",))
            if directive["directive"] == "STOP":
                return self._finish(state, directive["reason"])
            if directive["directive"] != "CODE":
                raise ValueError(f"Optimizer returned {directive}")
            if Path(str(directive["op_file"])).resolve() != panko.kernel(root):
                raise ValueError("CODE named a different implementation file")
            if not isinstance(directive["action_id"], str) or not directive["action_id"].strip():
                raise ValueError("CODE needs a harness action_id")
            if not isinstance(directive["intent"], str) or not directive["intent"].strip():
                raise ValueError("CODE needs a concrete intent")
            if not isinstance(directive.get("coder", True), bool):
                raise ValueError("CODE coder flag must be boolean")
            tree = panko.check_action(root, state, directive["action_id"])
            state.update(action_id=directive["action_id"], intent=directive["intent"],
                         phase="code" if directive.get("coder", True) else "step")
            panko.checkpoint(root, state)
            return {**self._next(state), "Frontier": self._frontier(tree, directive)}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            return self._report(False, str(exc))

    @action(consumes={"Source": 1}, produces={"Tune": 1, "Code": 1, "Report": 1})
    def apply(self, token: str) -> dict:
        root = Path(self.workspace)
        try:
            state = json.loads((root / "dispatch.json").read_text())
            if state["phase"] != "code":
                raise ValueError("No pending coder action")
            source = Path(str(token)).read_bytes()
            compile(source, "candidate.py", "exec")
            panko.write(root / "pending.py", source)
            state["phase"] = "publish"
            panko.checkpoint(root, state)
            return self._next(state)
        except (ValueError, SyntaxError, OSError, KeyError, TypeError) as exc:
            return self._report(False, str(exc))
