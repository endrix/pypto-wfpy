#!/usr/bin/env python3
# coding: utf-8
"""Node factory: `make_join(pool)` -- collect one result per worker slot."""

from __future__ import annotations

from typing import Any

from src.lib.records import RunResult
from src.lib.runner import LOCK
from wfpy import Port, action, task


def make_join(pool: int) -> type:
    """Build a join with one INPUT port + one @action per worker slot.

    Several `connect()` calls into ONE input port do not fan in -- the consumer
    fires once and the other tokens are stranded with no diagnostic. One port and
    one action per source is the idiom that actually collects every result.
    """
    ports: dict[str, Any] = {f"R{i}": Port[RunResult](direction="in") for i in range(pool)}
    ports["Results"] = Port[list](direction="out")

    def _mk(i: int) -> Any:
        def take(self: Any, result: RunResult) -> dict:
            # A module-level Lock, not a state field: task state is snapshotted
            # per firing for checkpointing and a Lock is not picklable.
            with LOCK:
                self._seen = list(self._seen) + [result]
                snapshot = list(self._seen)
            return {"Results": snapshot}

        take.__name__ = f"take_{i}"
        return action(consumes={f"R{i}": 1}, produces={"Results": 1})(take)

    ns: dict[str, Any] = {
        "__doc__": (
            "Accumulate per-slot results. Emits the running snapshot on every "
            "arrival; the summary reports on whichever snapshot is largest."
        ),
        "Ports": type("Ports", (), ports),
        "__annotations__": {"_seen": list},
        "_seen": [],
        **{f"take_{i}": _mk(i) for i in range(pool)},
    }
    return task(type("Join", (), ns))
