#!/usr/bin/env python3
# coding: utf-8
"""Validate several PyPTO kernels concurrently, in one wfpy graph.

Phase 1 runs one kernel per graph. This runs a pool of them at once, which is
where the actor model actually earns its keep: one actor never runs concurrently
with *itself*, so real parallelism needs one worker chain per slot.

    "Kernels" --> Dispatch --+--> Prepare0 --> RunSim0 --+--> Join --> Summary
                             +--> Prepare1 --> RunSim1 --+
                             +--> Prepare2 --> RunSim2 --+

Two wfpy constraints shape this, and both are load-bearing:

- An output port emits exactly ONE token per firing, so a runtime-sized kernel
  list can only reach a statically-sized pool through one port per slot. Hence
  `make_dispatch(pool)` builds the class at import time.
- Several `connect()` calls into ONE input port do not fan in -- the consumer
  fires once and the rest are stranded with no diagnostic. Hence
  `make_join(pool)` builds one input port plus one @action per slot.

The per-slot chain is Prepare -> RunSim with no Gate: the pass/fail split is
done once in Summary. Phase 1 already demonstrates Gate routing, and a Gate here
would need 2 ports per slot in the join for no added signal.

Run:
    ./run.sh pool
    ./run.sh pool examples/01_fused_elementwise.py examples/02_row_softmax.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nodes import LOCK, WF_OUT, RunResult, default_run_mode, execute_job, make_job  # noqa: E402
from wfpy import Port, action, connect, run, task, workflow  # noqa: E402

DEFAULT_KERNELS = [
    "examples/01_fused_elementwise.py",
    "examples/02_row_softmax.py",
    "examples/03_fused_linear.py",
    "examples/04_layer_norm.py",
]
POOL = int(__import__("os").environ.get("PYPTO_POOL", "4"))


def make_dispatch(pool: int) -> type:
    """Build a dispatcher with one output port per worker slot."""
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


def make_worker(slot: int) -> type:
    """Build one worker slot: resolve the kernel, then run it."""

    def work(self: Any, kernel: str) -> RunResult:
        job = make_job(kernel, self.run_mode, suffix=f"_{self.run_mode}", slot=slot)
        return execute_job(job, self.timeout_s)

    ns: dict[str, Any] = {
        "__doc__": f"Worker slot {slot}: prepare and run one kernel.",
        "Ports": type("Ports", (), {
            "Kernel": Port[str](direction="in"),
            "Result": Port[RunResult](direction="out"),
        }),
        "__annotations__": {"run_mode": str, "timeout_s": int},
        # run_mode has no default -> constructor parameter.
        # timeout_s has one     -> actor state.
        "timeout_s": 900,
        "work": action(consumes={"Kernel": 1}, produces={"Result": 1})(work),
    }
    return task(type(f"Worker{slot}", (), ns))


def make_join(pool: int) -> type:
    """Build a join with one input port + one @action per worker slot."""
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
            "arrival; Summary reports on whichever snapshot is the largest."
        ),
        "Ports": type("Ports", (), ports),
        "__annotations__": {"_seen": list},
        "_seen": [],
        **{f"take_{i}": _mk(i) for i in range(pool)},
    }
    return task(type("Join", (), ns))


@task
class Summary:
    """Report on the largest snapshot seen, and split pass from fail."""

    class Ports:
        Results = Port[list](direction="in")
        Out = Port[str](direction="out")

    # State: the join emits a snapshot per arrival, so keep the fullest one.
    _best: list = []

    @action(consumes={"Results": 1}, produces={"Out": 1})
    def summarize(self, results: list) -> str:
        with LOCK:
            if len(results) >= len(self._best):
                self._best = list(results)
            best = list(self._best)

        lines = []
        for r in sorted(best, key=lambda x: x.job.slot):
            where = f"slot {r.job.slot}, {r.job.run_mode}"
            if r.job.run_mode == "npu":
                where += f" dev{r.job.device_id}"
            if r.passed:
                lines.append(f"  PASS  {r.job.name:<22} {r.seconds}s  ({where})")
            else:
                lines.append(f"  FAIL  {r.job.name:<22} rc={r.rc}  {r.error[:60]}")
        passed = sum(1 for r in best if r.passed)
        wall = max((r.seconds for r in best), default=0.0)
        serial = round(sum(r.seconds for r in best), 1)
        head = (
            f"{passed}/{len(best)} kernels passed  "
            f"(slowest slot {wall}s vs {serial}s if run serially)"
        )
        return "\n".join([head, *lines])


Dispatch = make_dispatch(POOL)
Join = make_join(POOL)
Workers = [make_worker(i) for i in range(POOL)]


@workflow(inputs={"Kernels": list}, outputs={"Out": str})
def validate_pool() -> None:
    """Fan the kernel list across POOL worker chains, then join."""
    dispatch = Dispatch()
    join = Join()
    summary = Summary()
    workers = [Workers[i](run_mode=default_run_mode()) for i in range(POOL)]

    connect("Kernels", dispatch.Kernels)
    for i, worker in enumerate(workers):
        connect(getattr(dispatch, f"S{i}"), worker.Kernel)
        connect(worker.Result, getattr(join, f"R{i}"))
    connect(join.Results, summary.Results)
    connect(summary.Out, "Out")


def main() -> int:
    kernels = sys.argv[1:] or DEFAULT_KERNELS
    if len(kernels) > POOL:
        print(f"error: {len(kernels)} kernels but PYPTO_POOL={POOL}", file=sys.stderr)
        return 2
    result = run(validate_pool, inputs={"Kernels": kernels}, max_workers=POOL, out_dir=str(WF_OUT))
    lines = result["Out"] or []
    # The join emits a snapshot per arrival; the last one is the most complete.
    print(lines[-1] if lines else "(no output)")
    return 1 if not lines or "FAIL" in lines[-1] else 0


if __name__ == "__main__":
    sys.exit(main())
