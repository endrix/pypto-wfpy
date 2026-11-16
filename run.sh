#!/usr/bin/env bash
# Entry point for the pypto-wfpy workflows.
#
#   ./run.sh validate [kernel.py ...]   # phase 1: run + gate on the self-check
#   ./run.sh ir       [kernel.py ...]   # phase 2: per-pass IR diff report
#   ./run.sh plan     <workflow.py>     # dump the wfpy graph as JSON
#
# PYPTO_VENV overrides the interpreter; CANN_ENV overrides the CANN setup script.
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

cmd="${1:-validate}"
shift || true

case "$cmd" in
    validate) exec "$PY" "$ROOT/workflows/phase1_validate.py" "$@" ;;
    ir)       exec "$PY" "$ROOT/workflows/phase2_ir.py" "$@" ;;
    plan)
        wf="${1:?usage: ./run.sh plan <workflow.py>}"
        name="$([ "$(basename "$wf")" = "phase2_ir.py" ] && echo inspect_ir || echo validate_kernel)"
        exec "$PYPTO_VENV/bin/wfpy" plan "$wf" --workflow "$name" --format graph
        ;;
    *)
        echo "usage: ./run.sh {validate|ir|plan} [args...]" >&2
        exit 2
        ;;
esac
