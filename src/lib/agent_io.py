#!/usr/bin/env python3
# coding: utf-8
"""Read what an agent replied, whatever shape the port gave it.

Every `@agent` here declares its output format on the port -- `ext=".md"` for an
analysis, `ext=".json"` for anything a later node has to pick apart. That is not
decoration: for a `Port[File]` output wfpy writes the reply to
`<actor>__<port>__<n><ext>` and passes the **path** downstream
(`_agent_io_runtime.py:1125-1147`), and for a `.json` port it serialises a dict
or list with `json.dumps`. So a consumer receives a path, not text, and these
two helpers are the single place that knows it.

Why declare it at all, rather than taking a plain string:

- An undeclared reply lands as `.txt` and the prompt tells the agent nothing
  about the shape expected, so "reply with the complete file" competes with
  wfpy's own JSON envelope. A `.json` port makes the structure the contract
  instead of a request in prose.
- `ext` reaches the prompt: wfpy renders an output port as
  `Port (format: json)` (`_agent_io_runtime.py:285-287`), so the agent is told
  the format by the runtime as well as by the prompt.
- Code arriving inside a JSON field needs no fence stripping. A reply asked for
  as "the file and nothing else" routinely comes back fenced anyway, and every
  consumer then grew its own ```-stripping branch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_text(token: Any) -> str:
    """The reply as text, from a path token or a raw string.

    A token may be a `File` dataclass, a path string or the reply itself: a
    `Port[str]` passes the text through, while a `Port[File]` passes a path. Both
    reach here, so both are handled rather than asserted away.
    """
    raw = str(token)
    try:
        path = Path(raw)
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:
        pass
    return raw


def read_json(token: Any, required: tuple[str, ...] = ()) -> dict:
    """The reply parsed as a JSON object, with a readable error if it is not one.

    Raises `ValueError` rather than returning a half-empty dict. A malformed
    reply is a real failure and the loop around it should see it as one -- a
    silently empty field reads downstream as "the agent had nothing to say",
    which is how an empty payload once passed for a successful turn.
    """
    text = read_text(token).strip()
    if text.startswith("```"):
        # Fenced despite the prompt. Strip the fence rather than fail: the
        # content is usually right and the fence is a habit.
        lines = text.splitlines()
        text = "\n".join(lines[1: -1 if lines and lines[-1].strip() == "```" else len(lines)])

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply is not JSON ({exc.msg} at line {exc.lineno})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"reply is JSON but a {type(data).__name__}, not an object")

    missing = [key for key in required if not str(data.get(key, "")).strip()]
    if missing:
        raise ValueError(f"reply is missing {', '.join(missing)}")
    return data
