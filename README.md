# pypto-wfpy

PyPTO kernels driven as [wfpy](https://github.com/huawei-csl/wfpy) actor-dataflow
workflows: validate them, diff the compiler's IR pass by pass, find the limiting
hardware pipe, and repair failures with an agent.

Each phase is its own workflow, so any one can be swapped, fanned out or replaced
without touching the others.

| | Workflow | Graph | `run.sh` |
|---|---|---|---|
| **Phase 1** | `phase1_validate.py` | `Prepare → RunSim → Gate → Report` | `validate` |
| **Phase 1 (pool)** | `pool_validate.py` | `Dispatch → N×(Prepare→RunSim) → Join → Summary` | `pool` |
| **Phase 2** | `phase2_ir.py` | `PrepareIR → RunWithIR → CollectIR → DiffIR` | `ir` |
| **Phase 3** | `phase3_perf.py` | `Prepare → RunSim → CollectPerf → Analyze` | `perf` |
| **Phase 4** | `phase4_repair.py` | `Feed → Prepare → RunSim → Gate → Budget → Agent → ApplyPatch ↺` | `repair` |

```
examples/     three verified PyPTO Tensor-mode kernels (see examples/README.md)
workflows/    the wfpy graphs; _nodes.py holds the shared tokens and actors
tools/        run_with_ir.py -- enables per-pass IR dumping without editing a kernel
runs/         per-kernel work dirs: reports, diffs, simulator output
```

## Setup

PyPTO needs no build — CANN ships it. wfpy goes into the *same* venv so the
workflows and the kernels share one interpreter.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 -m venv --system-site-packages .venv
./.venv/bin/pip install numpy torch pyyaml matplotlib pandas
./.venv/bin/pip install -e /home/endrix/git/workflow/wfpy
```

`run.sh` sources the CANN env itself, so every command below works from a bare shell.

## Phase 1 — validate

Runs each kernel in the Soc CAModel simulator (CPU, no NPU required) and gates on
its self-check.

```console
$ ./run.sh validate
PASS  01_fused_elementwise   13.3s  (sim)
PASS  02_row_softmax         15.1s  (sim)
PASS  03_fused_linear        16.7s  (sim)
```

Failures route through the Gate's `Bad` port instead of crashing the run:

```console
$ ./run.sh validate runs/broken/01_wrong_math.py
FAIL  01_wrong_math          rc=1  fused_elementwise mismatch
```

### Concurrently, over a pool

`./run.sh pool` runs the kernels at the same time. One actor never runs
concurrently with *itself*, so real parallelism needs one worker chain per slot —
`PYPTO_POOL` sets how many (default 3).

```console
$ ./run.sh pool
3/3 kernels passed  (slowest slot 17.2s vs 45.5s if run serially)
  PASS  01_fused_elementwise   13.3s  (slot 0)
  PASS  02_row_softmax         15.0s  (slot 1)
  PASS  03_fused_linear        17.2s  (slot 2)
```

## Phase 2 — the compiler's IR, pass by pass

PyPTO lowers a kernel **TENSOR_GRAPH → TILE_GRAPH → BLOCK_GRAPH →
EXECUTE_GRAPH** across ~39 passes, dumping a readable `.tifwkgr` text graph both
before and after each one. Phase 2 pairs those up and diffs them.

```console
$ ./run.sh ir examples/03_fused_linear.py
PASS  03_fused_linear: 32/122 passes changed the graph;
      level shifts: TENSOR_GRAPH->TILE_GRAPH@ExpandFunction
      report: runs/03_fused_linear_ir/ir_report.md
      diffs:  runs/03_fused_linear_ir/ir_diffs
```

`ir_report.md` gives the level transitions, the level each function reaches, and
the passes that changed the graph most; `ir_diffs/*.diff` holds a unified diff
per changed pass. Dumps are source-attributed, so a line reads back to
`examples/03_fused_linear.py:48`.

Two things the reports make visible:

- **`Pass_05_ExpandFunction` is where tiling happens.** Its diff shows the
  64×128 tensor expanding into 32×64 tiles with `INCAST_LOCAL_BUF` /
  `OUTCAST_LOCAL_BUF` staging buffers.
- **`Pass_28_SubgraphToFunction` reveals the unit split.** The vector-only
  kernels produce **1** BLOCK_GRAPH leaf; `03_fused_linear` produces **2** — the
  cube and vector subgraphs. The diff also shows static shapes being
  parameterized into `RUNTIME_COA_GET_PARAM_*` runtime queries.

Dumping every pass costs ~15s and ~230 files per kernel. To narrow it:

```bash
./.venv/bin/python tools/run_with_ir.py examples/03_fused_linear.py \
    --passes ExpandFunction,SubgraphToFunction
```

## Phase 3 — which pipe is the bottleneck

The sim run already writes a MachineModel report, per-pipe usage and a tile-op
work profile. Phase 3 parses them and names the limiting pipe.

```console
$ ./run.sh perf
PASS  01_fused_elementwise     21,887 cycles  limited by MTE_IN (20.44%, memory-bound)
      next: MTE_OUT=1.88%, Vector=1.25%, Cube=0.0%
PASS  02_row_softmax           19,006 cycles  limited by MTE_IN (7.84%, memory-bound)
PASS  03_fused_linear          22,087 cycles  limited by MTE_IN (19.86%, memory-bound)
```

All three are memory-bound, not compute-bound. Two independent signals agree:
`MTE_IN` dominates the pipe ranking, and `ub-copyin` (77,824) dominates the
tile-op work profile while `matmul` is only 8,192. `Cube` reads 0% for the two
vector-only kernels, as it should.

## Phase 4 — bounded repair loop

A loop in wfpy is an ordinary feedback edge with no backpressure, so it **must**
terminate by itself. `Budget` is the only exit: a guarded action repairs while
attempts remain, an unguarded fallback gives up.

Transport defaults to `mock`, which makes termination observable offline — the
mock reply is deliberately not a fix, so the loop runs to its budget:

```console
$ ./run.sh repair runs/broken/01_wrong_math.py
[repair] transport=mock budget=2 skill=pypto-precision-debug
[repair] attempt 1/2 on 01_wrong_math: fused_elementwise mismatch
[repair] attempt 2/2 on attempt1: exited 0 but printed no PASS: line
GAVE UP   after 2 attempt(s); last tried attempt2 -- exited 0 but printed no PASS: line
```

For a real repair via the `claude` ACP connector (**this spends tokens**):

```bash
PYPTO_REPAIR_TRANSPORT=acp ./run.sh repair runs/broken/01_wrong_math.py
```

### cannbot skills

All 17 `cannbot-skills/ops/pypto-*` skills target Tensor mode, so they apply to
these kernels directly. The repair agent loads `pypto-precision-debug` (the right
one for a numeric mismatch); override with `PYPTO_REPAIR_SKILL`.

They are used **in place**, via `WF_SKILLS_ROOT`, which wfpy treats as its
highest-priority skill root. A symlink would not work: `_resolve_skill_dir()`
calls `.resolve()` and then requires the result to still sit under the root, so a
link pointing outside is skipped silently. `run.sh` exports the root for you.

## Notes on wfpy

Four behaviours worth knowing; the first two cost real debugging time.

- **A returned `dict` is a port map, not a payload.** With a single declared
  output port, wfpy does `result.get(port_name)` on a returned dict
  (`_action_runtime.py:269`), so a dict payload emits *nothing*, silently, and the
  run just reaches quiescence early. All tokens here are frozen dataclasses.
  `wf-out/<run>/run.wf-queues.json` is what diagnoses it — it lists every queue's
  size, so a producer that emitted nothing is obvious.
- **A field with a default is state, not a parameter.** `run_mode: str` is a
  constructor parameter; `run_mode: str = "sim"` becomes per-actor state and
  `Prepare(run_mode=...)` then raises. A parameter with a default is not
  expressible.
- **An input port does not fan in.** Extra `connect()` calls into one input port
  strand tokens with no diagnostic. Joining N sources needs N ports and one
  action each — hence the `make_join(pool)` factory, and `Feed` in phase 4.
- **An output port emits one token per firing.** A runtime-sized work list can
  only reach a statically-sized pool through one port per slot, so
  `make_dispatch(pool)` builds its class at import time.

Also: `RunSim` is a `@task`, not a `@tool`. A `@tool` raises on non-zero exit,
which aborts the whole graph instead of letting the Gate route the failure.

## Why Tensor mode

PyPTO **Pro** (`pypto_pro`) needs a full 0.3.0 source build *and* 950PR/950DT
silicon — `pypto_pro/runtime/platform.py` maps only `DAV_3510` and `DAV_9201`,
while this host reports `DAV_2201` (910B2). Tensor mode runs here today with no
build at all. See `examples/README.md`.
