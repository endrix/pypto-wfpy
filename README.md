# pypto-wfpy

PyPTO kernels driven as [wfpy](https://github.com/huawei-csl/wfpy) actor-dataflow
workflows, with the compiler's pass-by-pass IR exposed as a diff report.

Phases are separate workflows on purpose, so each can be swapped, fanned out or
replaced without touching the others.

| | Workflow | Graph |
|---|---|---|
| **Phase 1** | `workflows/phase1_validate.py` | `Prepare -> RunSim -> Gate -> Report` |
| **Phase 2** | `workflows/phase2_ir.py` | `PrepareIR -> RunWithIR -> CollectIR -> DiffIR` |

```
examples/     three verified PyPTO Tensor-mode kernels (see examples/README.md)
workflows/    the wfpy graphs
tools/        run_with_ir.py -- enables per-pass IR dumping without editing a kernel
runs/         per-kernel work dirs: IR reports, diffs, simulator output
```

## Setup

PyPTO needs no build — CANN ships it. wfpy goes into the *same* venv so the
workflow and the kernels share one interpreter.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 -m venv --system-site-packages .venv
./.venv/bin/pip install numpy torch pyyaml matplotlib pandas
./.venv/bin/pip install -e /home/endrix/git/workflow/wfpy
```

## Phase 1 — validate

Runs each kernel in the Soc CAModel simulator (CPU, no NPU required) and gates
on its self-check.

```bash
./run.sh validate                                    # all three kernels
./.venv/bin/python workflows/phase1_validate.py examples/02_row_softmax.py
```

```
PASS  01_fused_elementwise   13.5s  (sim)
PASS  02_row_softmax         15.2s  (sim)
PASS  03_fused_linear        17.6s  (sim)
```

Failures route through the Gate's `Bad` port rather than crashing the run, which
is where a repair loop attaches later:

```
FAIL  01_wrong_math          rc=1  fused_elementwise mismatch
```

## Phase 2 — inspect the compiler's IR

PyPTO lowers a kernel **TENSOR_GRAPH -> TILE_GRAPH -> BLOCK_GRAPH ->
EXECUTE_GRAPH** across ~39 passes, and dumps a readable `.tifwkgr` text graph
before and after each one. Phase 2 pairs those up and diffs them.

```bash
./run.sh ir                                          # all three kernels
./.venv/bin/python workflows/phase2_ir.py examples/03_fused_linear.py
```

```
PASS  03_fused_linear: 32/122 passes changed the graph;
      level shifts: TENSOR_GRAPH->TILE_GRAPH@ExpandFunction
      report: runs/03_fused_linear_ir/ir_report.md
      diffs:  runs/03_fused_linear_ir/ir_diffs
```

`ir_report.md` gives the level transitions, the IR level each function reaches,
and the passes that changed the graph most; `ir_diffs/*.diff` holds a unified
diff per changed pass. The dumps are source-attributed, so a line reads back to
`examples/03_fused_linear.py:48`.

Two things worth knowing from the reports:

- **`Pass_05_ExpandFunction` is where tiling happens.** Its diff shows the
  64x128 tensor expanding into 32x64 tiles with `INCAST_LOCAL_BUF` /
  `OUTCAST_LOCAL_BUF` staging buffers.
- **`Pass_28_SubgraphToFunction` reveals the unit split.** The vector-only
  kernels produce **1** BLOCK_GRAPH leaf; `03_fused_linear` produces **2** —
  the cube and vector subgraphs. The diff shows static shapes being
  parameterized into `RUNTIME_COA_GET_PARAM_*` runtime queries.

Dumping all passes costs ~15s and ~230 files per kernel. To narrow it:

```bash
./.venv/bin/python tools/run_with_ir.py examples/03_fused_linear.py \
    --passes ExpandFunction,SubgraphToFunction
```

## Notes on wfpy

Two behaviours cost real debugging time and are worth remembering:

- **A returned `dict` is a port map, not a payload.** With a single declared
  output port, wfpy does `result.get(port_name)` on a returned dict
  (`wfpy/_action_runtime.py:269`), so a dict payload emits *nothing*, silently.
  Both workflows use frozen dataclasses as tokens instead.
- **A field with a default is state, not a parameter.** `run_mode: str` is a
  constructor parameter; `run_mode: str = "sim"` would become per-actor state and
  `Prepare(run_mode=...)` would raise. A parameter with a default is not
  expressible.

Also: `RunSim` is a `@task`, not a `@tool`. A `@tool` raises on non-zero exit,
which aborts the whole graph instead of letting the Gate route the failure.

## Why Tensor mode

PyPTO **Pro** (`pypto_pro`) needs a full 0.3.0 source build *and* 950PR/950DT
silicon — `pypto_pro/runtime/platform.py` maps only `DAV_3510` and `DAV_9201`,
while this machine reports `DAV_2201` (910B2). Tensor mode runs here today with
no build at all. See `examples/README.md`.
