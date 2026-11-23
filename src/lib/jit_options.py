#!/usr/bin/env python3
# coding: utf-8
"""Add an options dict to a kernel's `@pypto.frontend.jit` decorator.

Two of the cannbot workflows turn a kernel into an instrumented one by adding a
dict to that decorator, and only that:

- `verify_options={"enable_pass_verify": True, "pass_verify_save_tensor": True,
  "pass_verify_pass_filter": []}` dumps every op's output tensor, which is what
  the precision triage compares.
- `debug_options={"runtime_debug_mode": 1}` collects swimlane and profiling
  data, which is what the tuning phases read.

Both are **decorator parameters**, not keys inside `runtime_options`. Putting
`debug_options` in `runtime_options` fails at compile time with
`key: runtime.debug_options does not exist` -- which is a clear error, but only
after a device run has been set up.

Writing an instrumented COPY rather than editing in place also satisfies
something the tuning orchestrator demands of a human: "⛔ 还原 debug_options",
restore the options when the phase ends. There is nothing to restore when the
original was never touched.
"""

from __future__ import annotations

import re

_JIT_RE = re.compile(r"@pypto\.frontend\.jit\((?P<args>[^)]*)\)")


def add_option(source: str, name: str, value: str, note: str = "") -> str:
    """Return `source` with `name=value` added to its jit decorator.

    The dict is bound to a module-level variable above the decorator rather than
    written inline, so the decorator line stays readable and the value can be
    found by eye in the instrumented copy.

    Raises if there is no jit decorator. A copy that silently instrumented
    nothing would run, produce no artifacts, and read downstream as "this kernel
    had nothing to report".
    """
    match = _JIT_RE.search(source)
    if match is None:
        raise ValueError("no @pypto.frontend.jit(...) decorator found")
    if name in match.group("args"):
        return source  # already instrumented

    args = match.group("args").strip()
    joined = f"{args}, {name}={name}" if args else f"{name}={name}"
    comment = f"# {note}\n" if note else ""
    return (
        source[: match.start()]
        + f"{comment}{name} = {value}\n\n\n@pypto.frontend.jit({joined})"
        + source[match.end():]
    )
