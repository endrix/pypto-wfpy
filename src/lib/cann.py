#!/usr/bin/env python3
# coding: utf-8
"""Make the CANN environment available without a wrapper shell.

PyPTO lives in CANN's own site-packages, which only reaches ``PYTHONPATH`` when
``set_env.sh`` is sourced. ``run.sh`` does that, but an IDE launching
``src/<workflow>.py`` directly does not, and the kernels run as subprocesses that
inherit the environment -- so every run fails with
``ModuleNotFoundError: No module named 'pypto'``.

This module sources that script once, in a subshell, and caches the result, so a
workflow started from an IDE behaves exactly like one started through run.sh.

``CANN_ENV`` overrides the script path; ``PYPTO_SKIP_CANN_ENV=1`` disables the
bootstrap entirely for an environment that is already prepared.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

CANN_ENV_SCRIPT = Path(os.environ.get("CANN_ENV", "/usr/local/Ascend/ascend-toolkit/set_env.sh"))

# Variables worth carrying over. set_env.sh also exports shell bookkeeping we do
# not want to inherit, so this is a allowlist rather than a wholesale replace.
_CARRY = (
    "PYTHONPATH", "LD_LIBRARY_PATH", "PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH", "TOOLCHAIN_HOME", "ASCEND_CUSTOM_OPP_PATH",
    "CMAKE_PREFIX_PATH", "ASCEND_ROOT_PATH",
)

_cached: dict[str, str] | None = None


def _sourced_environ() -> dict[str, str]:
    """Run set_env.sh in a subshell and read back the variables it exports."""
    # `set -u` is relaxed for the source: set_env.sh appends to PYTHONPATH and
    # LD_LIBRARY_PATH without checking whether they are set.
    script = (
        f'set +u; . "{CANN_ENV_SCRIPT}" >/dev/null 2>&1; '
        + "; ".join(f'printf "%s=%s\\0" {k} "${k}"' for k in _CARRY)
    )
    try:
        proc = subprocess.run(["bash", "-c", script], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out: dict[str, str] = {}
    for item in proc.stdout.decode("utf-8", errors="replace").split("\0"):
        if "=" in item:
            key, _, value = item.partition("=")
            if value:
                out[key] = value
    return out


def environ() -> dict[str, str]:
    """``os.environ`` with the CANN variables filled in, for subprocesses.

    Already-sourced environments are left alone: if ASCEND_TOOLKIT_HOME is set
    and CANN's site-packages is on PYTHONPATH, sourcing again would only
    duplicate path entries.
    """
    global _cached

    env = os.environ.copy()
    if os.environ.get("PYPTO_SKIP_CANN_ENV") == "1":
        return env
    if env.get("ASCEND_TOOLKIT_HOME") and "site-packages" in env.get("PYTHONPATH", ""):
        return env
    if not CANN_ENV_SCRIPT.is_file():
        print(
            f"[cann] warning: {CANN_ENV_SCRIPT} not found; pypto will probably not import. "
            "Set CANN_ENV to the right set_env.sh.",
            file=sys.stderr,
        )
        return env

    if _cached is None:
        _cached = _sourced_environ()
    env.update(_cached)
    return env


def apply_to_process() -> None:
    """Also put the CANN variables into *this* process's environment.

    Call it from a workflow's entry point so an IDE run matches a run.sh run,
    including for anything that imports pypto in-process rather than in a
    subprocess.
    """
    os.environ.update({k: v for k, v in environ().items() if k in _CARRY})
