#!/usr/bin/env python3
# coding: utf-8
"""Node factory: `make_dispatch(pool)` -- spread a kernel list across worker slots."""

from __future__ import annotations

from typing import Any

from wfpy import Port, action, task


def make_dispatch(pool: int) -> type:
    """Build a dispatcher with one OUTPUT port per worker slot.

    A factory rather than a plain class because a wfpy output port emits exactly
    ONE token per firing, so a runtime-sized kernel list can only reach a
    statically-sized pool through a static port per slot. The pool size is known
    at import time, so the ports can be generated then.
    """
    ports: dict[str, Any] = {"Kernels": Port[list](direction="in")}
    for i in range(pool):
        ports[f"S{i}"] = Port[str](direction="out")

    def fan(self: Any, kernels: list) -> dict:
        names = [str(k) for k in kernels]
        if len(names) > pool:
            # One firing emits one token per port, so extra kernels would be
            # silently dropped and the run would look complete when it is not.
            raise RuntimeError(
                f"{len(names)} kernels but only {pool} slots; set PYPTO_POOL>={len(names)}"
            )
        # Slots beyond the kernel count emit nothing, so their chains stay idle.
        return {f"S{i}": name for i, name in enumerate(names)}

    ns: dict[str, Any] = {
        "__doc__": "Spread the kernel list across the worker slots.",
        "Ports": type("Ports", (), ports),
        "fan": action(consumes={"Kernels": 1}, produces={f"S{i}": 1 for i in range(pool)})(fan),
    }
    return task(type("Dispatch", (), ns))
