"""Locate the cannbot ops collection used by the workflow agents and profiler."""

import os
from pathlib import Path

_DEFAULT_OPS = Path(__file__).resolve().parents[3] / "pypto" / "cannbot-skills" / "ops"


def ops_root() -> Path:
    # Tuning can append a second wfpy root for nested skills.
    roots = os.environ.get("WF_SKILLS_ROOT", "").split(os.pathsep)
    return Path(roots[0] or os.environ.get("CANNBOT_OPS") or _DEFAULT_OPS)
