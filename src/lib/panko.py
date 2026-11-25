"""Persist the wfpy dispatch alongside the vendor-owned PANKO search tree."""

from __future__ import annotations

from hashlib import sha256
import ast
import json
from pathlib import Path
import os
import shlex
import shutil
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory, mkdtemp

from src.lib import cann
from src.lib.records import JobSpec
from src.lib.runner import PYTHON, REPO_ROOT, execute_job
from src.lib.skill_paths import ops_root

OP = "kda_decay_matrix_codex"
MANUAL = "pypto-op-perf-panko-manual"
FIXTURE = REPO_ROOT / "tools/panko_kda_fixture.py"
DRIVER = REPO_ROOT / "tools/check_kda.py"


def write(path: Path, data: bytes) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
    Path(tmp.name).replace(path)


def checkpoint(root: Path, state: dict) -> None:
    write(root / "dispatch.json", json.dumps(state, indent=2).encode())


def kernel(root: Path) -> Path:
    return root / f"{OP}_impl.py"


def setup(root: Path, source: Path, device: int, budget: int, tile_search: bool) -> dict:
    template = source.read_bytes()
    manifest = {"source_sha256": sha256(template).hexdigest(), "device": device, "eval_budget": budget,
                "tile_search": tile_search, "fixture_sha256": sha256(FIXTURE.read_bytes()).hexdigest(),
                "driver_sha256": sha256(DRIVER.read_bytes()).hexdigest(),
                "lint_root": os.environ.get("PYPTO_PANKO_LINT_ROOT", "")}
    path = root / "dispatch.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["manifest"] != manifest:
            raise ValueError("Resume requires the original source, fixtures, device, budget and tile-search settings")
        if state["phase"] != "prepare":
            return state
    else:
        if any(p.name != ".wfpy.lock" for p in root.iterdir()):
            raise ValueError(f"Choose a new workspace or resume a directory with dispatch.json: {root}")
        state = {"manifest": manifest, "phase": "prepare"}
        checkpoint(root, state)
    marker = b"@pypto.frontend.jit("
    if template.count(marker) != 1 or b"debug_options=" in template:
        raise ValueError("Expected the verified KDA seed with one uninstrumented JIT entry")
    (root / "eval").mkdir(exist_ok=True)
    # The vendor imports this file outside the repository; supply its source root explicitly.
    (root / "eval/test_inputs.py").write_text(
        f"import sys\nsys.path.insert(0, {str(REPO_ROOT)!r})\nfrom tools.panko_kda_fixture import build_inputs\n")
    shutil.copyfile(FIXTURE, root / f"{OP}_golden.py")
    (root / "SPEC.md").write_text(
        "FP32 causal KDA decayed Gram matrix. Fixed seed 0. Exact masked zeros. "
        "Compare five fixtures against the FP64 reference with atol=rtol=1e-3. "
        "Measure 128x64 with the diagonal included.\n")
    write(kernel(root), template.replace(marker, marker + b'debug_options={"runtime_debug_mode": 1}, '))
    state["phase"] = "init"
    checkpoint(root, state)
    return state


def harness_root(root: Path) -> Path:
    """Prepare the installed harness for CANN enums and resident JSON replies."""
    installed = ops_root() / MANUAL
    local = root / ".panko-runtime"
    if local.is_dir():
        return local
    original = installed / "scripts/feasibility.py"
    if not original.is_file():
        return installed
    source = original.read_text()
    loop = "        for member in members:\n"
    protocol = installed / "scripts/jsonio.py"
    emit = "    _log.info(json.dumps(obj, ensure_ascii=False, indent=indent))"
    needs_redirect = protocol.is_file() and emit in protocol.read_text()
    harness_file = installed / "scripts/panko_harness.py"
    shell = "    shell = [n for n in tree.body if not _is_kernel(n)]"
    nested_seed = harness_file.is_file() and shell in harness_file.read_text()
    if loop not in source and not needs_redirect and not nested_seed:
        return installed
    # Keep the harness, catalog, resident workers and subprocess evaluator in
    # one revision. Publish the copy atomically so a resumed run can reuse it.
    with TemporaryDirectory(dir=root) as tmp:
        runtime = Path(tmp) / MANUAL
        shutil.copytree(installed, runtime, ignore=shutil.ignore_patterns("__pycache__"))
        (runtime / "scripts/feasibility.py").write_text(source.replace(
            loop, "        values = getattr(members, '__members__', None)\n"
                  "        for member in (values.values() if values is not None else members):\n", 1))
        # A resident harness redirects stdout to a new buffer for each command.
        # The installed protocol logger otherwise keeps the first command's buffer.
        if needs_redirect:
            (runtime / "scripts/jsonio.py").write_text(protocol.read_text().replace(
                emit, '    if "_handler" in globals():\n        _handler.stream = sys.stdout\n' + emit, 1))
        if nested_seed:
            # The verified seed chooses sim/NPU inside a cached factory. Freeze its
            # host logic, JIT signature and decorator, while allowing device body edits.
            adapter = """    for factory in tree.body:
        if isinstance(factory, ast.FunctionDef) and factory.name == "_kernel_for_mode":
            for child in factory.body:
                if _is_kernel(child) and child.name == "kda_decay_matrix_codex_kernel":
                    child.body = [ast.Pass()]
"""
            (runtime / "scripts/panko_harness.py").write_text(
                harness_file.read_text().replace(shell, adapter + shell, 1))
        version = sha256()
        for file in sorted(p for p in runtime.rglob("*") if p.is_file()):
            version.update(str(file.relative_to(runtime)).encode() + b"\0")
            version.update(sha256(file.read_bytes()).digest())
        (runtime / "scripts/VERSION").write_text(f"sha256:{version.hexdigest()}\n")
        runtime.rename(local)
    return local


def close_workers(root: Path) -> None:
    opt = root / "optimization"
    if any((opt / name / "active.json").exists() for name in (".harness", ".evaluation")):
        subprocess.run([PYTHON, str(harness_root(root) / "scripts/harness_session.py"),
                        "--op-dir", str(root), "--close"], check=True, timeout=40,
                       env=cann.environ(), capture_output=True)


def check_action(root: Path, state: dict, action_id: str) -> dict:
    tree = json.loads((root / "optimization/search_state.json").read_text())
    if (tree["op"] != OP or int(tree["device_id"]) != state["manifest"]["device"]
            or tree["config"]["eval_budget"] != state["manifest"]["eval_budget"]
            or tree["progress"]["stop_reason"]
            or not any(n["id"] == action_id for n in tree["nodes"]["open"])):
        raise ValueError(f"Action {action_id} is not open in this campaign")
    return tree


def request(root: Path, state: dict) -> str:
    config = state["manifest"]
    runtime = harness_root(root)
    test = f"TILE_FWK_DEVICE_ID={config['device']} " + shlex.join([
        PYTHON, str(DRIVER), "verify", str(kernel(root)), str(root / f"{OP}_golden.py")])
    env = {"TILE_FWK_DEVICE_ID": str(config["device"]),
           "CANN_ENV": os.environ.get("CANN_ENV", "/usr/local/Ascend/ascend-toolkit/set_env.sh")}
    if runtime.parent == root:
        env["GIT_CEILING_DIRECTORIES"] = str(runtime)

    return json.dumps({
        "phase": state["phase"], "op": OP, "op_dir": str(root), "op_file": str(kernel(root)),
        "device": config["device"], "test_command": test,
        "config": {"eval_budget": config["eval_budget"], "bayesian_optimization_tile_search": config["tile_search"]},
        "P_ref": 0.001, "python": PYTHON, "skill_root": str(runtime),
        "lint_root": config["lint_root"] or None,
        "candidate_file": str(kernel(root)) if state["phase"] == "step" else None,
        "action_id": state.get("action_id"),
        "environment": env,
        "instructions": "Run vendor scripts from skill_root with the supplied environment. Source CANN_ENV. "
        "The editable device body is kda_decay_matrix_codex_kernel inside _kernel_for_mode. Module-level "
        "helpers, constants, the host wrapper and the nested function signature and decorator are frozen. "
        "Use the existing state if INIT or STEP was interrupted. Reconcile the action with "
        "search_state.json and the harness trajectory before issuing commands. Do not re-initialize or "
        "record an already recorded evaluation. Use harness_session.py so STOP publishes controller-terminal.json. "
        "Check harness stop immediately after init and before issuing more edits when the evaluation budget is reached. "
        "When lint_root is null use the fixed KDA test command and wrapper feasibility gate; report the absence "
        "of the optional vendor layout lint. Use the specified interpreter and run the CANN setup when needed. "
        "Use exactly one directive. Report infrastructure or prerequisite failures via an ERROR directive with reason."
    }, indent=2)


def terminal(root: Path, state: dict) -> dict:
    opt = root / "optimization"
    tree = json.loads((opt / "search_state.json").read_text())
    marker = json.loads((opt / "controller-terminal.json").read_text())
    prog = tree["progress"]
    best = next((n for n in tree["nodes"]["closed"] if n["id"] == tree["best_node"]), {})
    digest = sha256(kernel(root).read_bytes()).hexdigest()
    if not (tree["op"] == OP and int(tree["device_id"]) == state["manifest"]["device"]
            and tree["config"]["eval_budget"] == state["manifest"]["eval_budget"]
            and tree["config"]["bayesian_optimization_tile_search"] == state["manifest"]["tile_search"]
            and marker["directive"] == "STOP" and marker["reason"] == prog["stop_reason"]
            and marker["run_start"] is not None and marker["run_start"] == prog["_start_ts"]
            and marker["delivered_code_hash"] == digest == best.get("code_hash")
            and best.get("eval", {}).get("s") == 1 and best["eval"]["p"] == prog["best_latency_us"]
            and prog["best_latency_us"] > 0 and 0 < prog["evals_used"] <= state["manifest"]["eval_budget"]):
        raise ValueError("The harness terminal marker, best measured node and delivered file do not match")
    if prog["stop_reason"] in ("harness_fault", "device_unavailable", "preopt_incorrect"):
        raise ValueError(f"PANKO stopped with {prog['stop_reason']}")
    return prog


def without_profiling(source: bytes) -> bytes:
    """Remove the profiler flag in the delivered JIT, leaving the vendor's winner intact."""
    modes = []
    for call in ast.walk(ast.parse(source)):
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != "pypto.frontend.jit":
            continue
        for kw in call.keywords:
            if kw.arg == "debug_options" and isinstance(kw.value, ast.Dict):
                for key, value in zip(kw.value.keys, kw.value.values):
                    if isinstance(key, ast.Constant) and key.value == "runtime_debug_mode":
                        modes.append(value)
    if len(modes) != 1 or not isinstance(modes[0], ast.Constant) or modes[0].value != 1:
        raise ValueError("Cannot identify the single measured JIT's runtime_debug_mode=1")
    mode = modes[0]
    lines = source.splitlines(keepends=True)
    index = mode.lineno - 1
    lines[index] = lines[index][:mode.col_offset] + b"0" + lines[index][mode.end_col_offset:]
    return b"".join(lines)


def verify_winner(root: Path, state: dict) -> str:
    directory = mkdtemp(prefix="final-", dir=root)
    path = Path(directory) / "winner.py"
    write(path, without_profiling(kernel(root).read_bytes()))
    job = JobSpec(str(path), OP, True, "npu", directory,
                  (PYTHON, str(DRIVER), "final", str(path), str(root / f"{OP}_golden.py")),
                  device_id=state["manifest"]["device"], phase="final")
    run = execute_job(job, int(os.environ.get("PYPTO_KDA_FINAL_TIMEOUT_S", 1800)))
    if not run.passed:
        raise ValueError(f"Delivered kernel failed the final NPU check: {run.error}. Logs: {directory}")
    write(root / "winner.py", path.read_bytes())
    return directory
