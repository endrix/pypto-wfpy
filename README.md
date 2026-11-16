# pypto-wfpy

PyPTO kernels driven as [wfpy](https://github.com/huawei-csl/wfpy) actor-dataflow
workflows: validate them, diff the compiler's IR pass by pass, find the limiting
hardware pipe, and repair failures with an agent.

Each phase is its own workflow, so any one can be swapped, fanned out or replaced
without touching the others.

| | Workflow | Graph | `run.sh` |
|---|---|---|---|
| **Phase 1** | `workflows/validate.py` | `Prepare → RunSim → Gate → VerdictReport` | `validate` |
| **Phase 1 (pool)** | `workflows/pool.py` | `Dispatch → N×Worker → Join → PoolSummary` | `pool` |
| **Phase 2** | `workflows/ir.py` | `PrepareIR → RunWithIR → CollectIR → DiffIR` | `ir` |
| **Phase 3** | `workflows/perf.py` | `Prepare → RunSim → CollectPerf → AnalyzePerf` | `perf` |
| **Phase 4** | `workflows/repair.py` | `RunKernel → Decide → RepairAgent → ApplyPatch ↺ → RepairReport` | `repair` |

```
examples/           four verified PyPTO Tensor-mode kernels (see examples/README.md)
fixtures/           deliberately broken kernels, for testing the repair loop
src/validate.py     the five top-level workflows live at the src/ root
src/pool.py
src/ir.py
src/perf.py
src/repair.py
src/workflows/      sub-workflows, when a workflow is instantiated inside another
src/tasks/          one @task per file
src/agents/         one @agent per file
src/tools/          one @tool per file
src/lib/            shared non-actor code: records, runner, graph, sim_report,
                    repair_config
tools/              run_with_ir.py -- per-pass IR dumping without editing a kernel
runs/               per-kernel work dirs: reports, diffs, simulator output
wf-out/             wfpy run dirs: journals, queue traces, agent transcripts
```

**One actor per file.** Every `@task`, `@agent` and `@tool` lives in its own
module, and a workflow file contains only its `@workflow` wiring plus a CLI
entry. This mirrors `wfpy-skillyard`, keeps a node diffable on its own, and makes
a graph's shape readable from the directory listing.

**The `src/` root holds workflows, not library code.** A file at `src/*.py` is an
entry point you can run. Shared non-actor code -- dataclasses, path handling,
parsers, config -- lives in `src/lib/`, so the root stays a list of what this
project *does*. Sub-workflows (a `@workflow` instantiated inside another) go in
`src/workflows/`.

Because wfpy loads a workflow by path, each one puts the project root on
`sys.path` before its `src.*` imports:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

Both `runs/` and `wf-out/` are always inside the repo, whatever directory you
invoke from. wfpy defaults its `out_dir` to `./wf-out` relative to the
*invocation* cwd, which would otherwise scatter run logs; every workflow pins it
to `<repo>/wf-out` instead, overridable with `PYPTO_WF_OUT`. `./run.sh wfpy run
...` passes `--out-dir` for the same reason, since the bare `wfpy` CLI bypasses
the workflows' own `main()`.

## Setup

PyPTO needs no build — CANN ships it. wfpy goes into the *same* venv so the
workflows and the kernels share one interpreter.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 -m venv --system-site-packages .venv
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
./.venv/bin/pip install torch_npu==2.9.0          # device runs; omit for sim only
./.venv/bin/pip install numpy pyyaml matplotlib pandas
./.venv/bin/pip install -e "/home/endrix/git/workflow/wfpy[agent]"
```

> **Pin the torch/torch_npu pair; do not install "latest".** 2.9.0/2.9.0 works
> here (8 devices visible), as does 2.10.0/2.10.0. But `torch_npu` 2.13.0 and
> 2.14.0 each abort at import with a C++ `Duplicated key
> 'pinned_reserve_segment_size_mb'`, and because modern torch auto-loads the NPU
> backend through an entry point, a broken `torch_npu` breaks plain
> `import torch` and therefore *simulator* runs too.
>
> Use the CPU index for torch: the default index serves a `+cu130` build that
> drags ~3GB of unusable `nvidia-*` wheels onto an Ascend box.
>
> Other venvs on this machine already run the working pair and are the reference
> if these versions ever stop resolving: `pto-kernels/.venv`,
> `pto-mlir-cannbench/.venv` (2.9.0) and `workflow/old_wfpy/.venv` (2.10.0).

`run.sh` sources the CANN env itself, so every command below works from a bare shell.

### Run mode: device or simulator

The workflows auto-detect. `default_run_mode()` returns `npu` when `torch_npu`
imports, else falls back to `sim`; `PYPTO_RUN_MODE=npu|sim` forces either.

- **`npu`** runs on the real hardware, and is verified: all four kernels pass on
  910B2, spread over devices 0-3. Tensor mode fully supports this host — the
  Pro-mode arch limit below does *not* apply.
- **`sim`** is the Soc CAModel on CPU. No device needed, but its cycle counts are
  a **cost model, not a measurement**.

Note that `perf` (phase 3) only works in `sim`: a device run writes compiled
kernels and no performance data, so the phase reports `SKIP`. Real device
profiling is reachable — `pypto.set_debug_options(runtime_debug_mode=1)` emits
`tilefwk_L1_prof_data.json` (per-core task start/end, in ns),
`merged_swimlane.json` and `bubble_analysis.log` — but it is a different shape
from the simulator's report: task timings and waits, not per-pipe utilisation.
So it cannot directly confirm or refute the simulator's `MTE_IN`-bound verdict;
that would need msprof-level profiling. Nothing reads it yet.

Concurrent slots get one device each via `TILE_FWK_DEVICE_ID`
(`PYPTO_DEVICES`, default `0,1,2,3,4,5,6,7`). Without that, pooled runs would
serialise on device 0 and the measured speed-up would be an artefact.

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

> **These figures come from the simulator**, so they are a cost model's
> prediction rather than a hardware measurement. Treat the *ranking* as the
> signal and the absolute cycle counts as indicative. Run with
> `PYPTO_RUN_MODE=npu` for real timings, and expect them to differ.

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
./.venv/bin/pip install -e "/home/endrix/git/workflow/wfpy[agent]"   # once
PYPTO_REPAIR_TRANSPORT=acp PYPTO_REPAIR_BUDGET=1 \
    ./run.sh repair runs/broken/01_wrong_math.py
```

The `[agent]` extra is required: without it the ACP transport reaches
`_invoke_agent_opencode_acp` and dies on `ModuleNotFoundError: No module named
'acp'`. `PYPTO_REPAIR_BUDGET=1` caps the run at a single agent call.

Two logging traps when checking whether a live agent really ran:

- The banner always prints `provider=openai, model=gpt-4o`, whatever the
  transport. It is wfpy's default model metadata, not evidence of the route
  taken.
- `wfpy plan` rewrites this file's `@agent` decorator into literals (see the
  warning comment above it). A previous commit shipped a hardcoded
  `transport="mock"` while the banner still printed `TRANSPORT`, so
  `PYPTO_REPAIR_TRANSPORT=acp` reported the agent transport and ran the mock.
  Re-read the decorator after any `./run.sh plan`.

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

This is a limit on **Pro mode only**, and is unrelated to the device/simulator
choice above: Tensor mode runs on these 910B2s natively. If a workflow reports
`sim`, that means `torch_npu` is missing or `PYPTO_RUN_MODE=sim` is set — not
that the hardware is unsupported.
