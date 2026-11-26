#!/usr/bin/env python3
# coding: utf-8
"""Node factory: `make_dispatch(pool)` -- spread a kernel list across worker slots."""

from __future__ import annotations

from typing import Any

from wfpy import Port, action, task


def make_dispatch(pool: int) -> type:
    """Build a dispatcher with one OUTPUT port per worker slot.

    A single list token per active slot carries its round-robin batch.
    Each worker executes its list sequentially while the other slots run.
    """
    ports: dict[str, Any] = {"Kernels": Port[list](direction="in")}
    for i in range(pool):
        ports[f"S{i}"] = Port[list](direction="out")

    def fan(self: Any, kernels: list) -> dict:
        names = [str(k) for k in kernels]
        return {f"S{i}": names[i::pool] for i in range(min(pool, len(names)))}

    ns: dict[str, Any] = {
        "__doc__": "Spread the kernel list across the worker slots.",
        "Ports": type("Ports", (), ports),
        "fan": action(consumes={"Kernels": 1}, produces={f"S{i}": 1 for i in range(pool)})(fan),
    }
    return task(type("Dispatch", (), ns))
