#!/usr/bin/env bash
# Entry point for the pypto-wfpy workflows.
#
#   ./run.sh validate [kernel.py ...]   # phase 1: run + gate on the self-check
#   ./run.sh pool     [kernel.py ...]   # phase 1, all kernels concurrently
#   ./run.sh ir       [kernel.py ...]   # phase 2: per-pass IR diff report
#   ./run.sh perf     [kernel.py ...]   # phase 3: bottleneck from sim counters
#   ./run.sh repair   <kernel.py>       # phase 4: bounded agent repair loop
#   ./run.sh repair-softmax [kernel.py] # repair from a file source, ends in a viewer
#   ./run.sh sim-analyze [kernel.py]    # simulate -> analyse -> visualise (nested)
#   ./run.sh optimize [kernel.py]       # phase 5: bounded agentic perf search
#   ./run.sh triage [suspect.py] [reference.py]  # localise a bad op (NPU only)
#   ./run.sh profile [kernel.py]        # measured device profile (NPU only)
#   ./run.sh author [torch.py] [op]      # author a kernel from PyTorch (5 agents)
#   ./run.sh kda [torch.py] [op]         # author + optimise + redesign (Codex ACP by default)
#   ./run.sh plan     <workflow.py>     # dump the wfpy graph as JSON
#   ./run.sh wfpy     run <args...>    # raw wfpy passthrough, wf-out pinned
#
# PYPTO_VENV overrides the interpreter; CANN_ENV overrides the CANN setup script.
# PYPTO_POOL sets the concurrent slot count (default 4).
# PYPTO_RUN_MODE=npu|sim forces the run mode; PYPTO_DEVICES spreads pool slots.
# PYPTO_WF_OUT overrides the run directory (default <repo>/wf-out).
# PYPTO_REPAIR_TRANSPORT=acp uses the default Claude connector (spends tokens);
# PYPTO_REPAIR_BUDGET caps the repair attempts.
# WF_SKILLS_ROOT points wfpy at the cannbot pypto-* skills.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
PYPTO_VENV="${PYPTO_VENV:-$ROOT/.venv}"
PY="$PYPTO_VENV/bin/python"

if [ ! -x "$PY" ]; then
    echo "error: no interpreter at $PY" >&2
    echo "       create it with the steps in README.md, or set PYPTO_VENV." >&2
    exit 1
fi

# PyPTO lives in CANN's site-packages, which set_env.sh puts on PYTHONPATH.
if [ -f "$CANN_ENV" ]; then
    # set_env.sh appends to PYTHONPATH/LD_LIBRARY_PATH without checking whether
    # they are set, so `set -u` would make it spew "unbound variable". Relax -u
    # for the source only, then restore it.
    set +u
    # shellcheck disable=SC1090
    source "$CANN_ENV"
    set -u
else
    echo "warning: $CANN_ENV not found; pypto may not be importable" >&2
fi

# The cannbot pypto-* skills are used in place; a symlink into them would be
# rejected by wfpy's skill-root containment check.
CANNBOT_OPS="${CANNBOT_OPS:-$ROOT/../pypto/cannbot-skills/ops}"
if [ -z "${WF_SKILLS_ROOT:-}" ] && [ -d "$CANNBOT_OPS" ]; then
    export WF_SKILLS_ROOT="$CANNBOT_OPS"
fi

cmd="${1:-validate}"
shift || true

case "$cmd" in
    validate) exec "$PY" "$ROOT/src/validate.py" "$@" ;;
    pool)     exec "$PY" "$ROOT/src/pool.py" "$@" ;;
    ir)       exec "$PY" "$ROOT/src/ir.py" "$@" ;;
    perf)     exec "$PY" "$ROOT/src/perf.py" "$@" ;;
    repair)   exec "$PY" "$ROOT/src/repair.py" "$@" ;;
    repair-softmax) exec "$PY" "$ROOT/src/repair_softmax.py" "$@" ;;
    sim-analyze)    exec "$PY" "$ROOT/src/sim_analyze.py" "$@" ;;
    optimize)       exec "$PY" "$ROOT/src/optimize.py" "$@" ;;
    triage)         exec "$PY" "$ROOT/src/precision_triage.py" "$@" ;;
    profile)        exec "$PY" "$ROOT/src/profile_kernel.py" "$@" ;;
    author)         exec "$PY" "$ROOT/src/new_kernel.py" "$@" ;;
    kda)            exec "$PY" "$ROOT/src/kda_pipeline.py" "$@" ;;
    plan)
        wf="${1:?usage: ./run.sh plan <workflow.py>}"
        case "$(basename "$wf")" in
            ir.py)      name=inspect_ir ;;
            perf.py)    name=analyze_perf ;;
            repair.py)          name=repair_kernel ;;
            repair_softmax.py)  name=repair_softmax ;;
            sim_analyze.py)     name=sim_analyze ;;
            optimize.py)        name=optimize_kernel ;;
            precision_triage.py) name=precision_triage ;;
            profile_kernel.py)  name=profile_kernel ;;
            new_kernel.py)      name=new_kernel ;;
            kda_pipeline.py)    name=kda_pipeline ;;
            pool.py)    name=validate_pool ;;
            *)          name=validate_kernel ;;
        esac
        exec "$PYPTO_VENV/bin/wfpy" plan "$wf" --workflow "$name" --format graph "${@:2}"
        ;;
    wfpy)
        # Raw `wfpy run` passthrough, with the run dir pinned the same way the
        # workflows' own main() functions pin it.
        [[ "${1:-}" != run ]] || shift
        exec "$PYPTO_VENV/bin/wfpy" run --out-dir "${PYPTO_WF_OUT:-$ROOT/wf-out}" "$@"
        ;;
    *)
        echo "usage: ./run.sh {validate|pool|ir|perf|repair|repair-softmax|sim-analyze|optimize|triage|profile|author|kda|plan|wfpy} [args...]" >&2
        exit 2
        ;;
esac
