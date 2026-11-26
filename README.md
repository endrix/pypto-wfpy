# pypto-wfpy

PyPTO kernels driven as [wfpy](https://github.com/huawei-csl/wfpy) actor-dataflow
workflows: validate them, diff the compiler's IR pass by pass, find the limiting
hardware pipe, and repair failures with an agent.

Each phase is its own workflow, so any one can be swapped, fanned out or replaced
without touching the others.

| | Workflow | Graph | `run.sh` |
|---|---|---|---|
| **Phase 1** | `src/validate.py` | `Prepare → RunSim → Gate → VerdictReport` | `validate` |
| **Phase 1 (pool)** | `src/pool.py` | `Dispatch → N×Worker → Join → PoolSummary` | `pool` |
| **Phase 2** | `src/ir.py` | `PrepareIR → RunWithIR → CollectIR → DiffIR` | `ir` |
| **Phase 3** | `src/perf.py` | `Prepare → RunSim → CollectPerf → AnalyzePerf` | `perf` |
| **Phase 4** | `src/repair.py` | `repair_flow` (`RunKernel → Decide → RepairAgent → ApplyPatch ↺ → WriteRepairReport`) | `repair` |
| **Phase 4 (nested)** | `src/repair_softmax.py` | `KernelSource → repair_flow → RepairReportViewer` | `repair-softmax` |
| **Phase 5** | `src/sim_analyze.py` | `KernelSource → simulate_flow → analyze_flow → visualize_flow` | `sim-analyze` |
| **Phase 6** | `src/optimize.py` | `KernelSource → optimize_flow → MarkdownViewer` | `optimize` |
| Device profile | `src/profile_kernel.py` | Capture and analyze an NPU timeline | `profile` |
| Precision triage | `src/precision_triage.py` | Compare suspect and reference on the NPU | `triage` |
| Author | `src/new_kernel.py` | Turn a PyTorch source into a PyPTO kernel | `author` |
| KDA | `src/kda_pipeline.py` | Author, verify, optimize, then run the selected kernel on the NPU | `kda` |
| NPU-guided tuning | `src/npu_tune.py` | Profile the incumbent, edit it, verify on device and compare paired timings | `npu-tune` |
| Competing kernel designs | `src/residual_rmsnorm.py` | Generate two residual RMSNorm kernels, tune each and compare with unfused PyTorch | `residual-rmsnorm` |
| Precision repair | `src/precision_repair.py` | Localize a real attention fault, patch one statement and rerun the failing fixtures | `precision-repair` |
| PANKO search | `src/panko_search.py` | Resume an optimization tree and verify the delivered KDA kernel | `panko` |

## KDA workflow

Open `src/kda_pipeline.py` in the wfpy IDE or use the same workflow from the CLI.
It takes `torch_src/kda_decay_matrix.py` as input. All ten agents use the
**Codex ACP connector and `gpt-6.2-sol`** by default and write their replies and
artifacts in English. To change the model, set `PYPTO_COMBO_MODEL` to a model
offered by your connector. To switch connectors, set both variables.

```bash
PYPTO_WF_OUT="$PWD/src/wf-out" ./run.sh kda
PYPTO_COMBO_MODEL="<model-id>" ./run.sh kda
PYPTO_COMBO_CONNECTOR=claude PYPTO_COMBO_MODEL=sonnet ./run.sh kda
./run.sh plan src/kda_pipeline.py --hierarchy
```

Set these variables in the IDE process environment before opening the workflow.
The CLI writes journals to `wf-out/` by default. For the same output location
as the first command, select `src/wf-out` in the IDE's run output setting.
Standalone authoring and KDA use a fresh workspace per run. Each KDA run keeps its authoring documents, candidates and evaluation logs under
`custom/kda_decay_matrix/<workspace-id>/`. In the IDE, the `combo` graph has a viewer connected to the author’s
`Verdict` output. Its `search` graph has viewers for the analysis node’s
`Summary`, `PerfReport` and `BubbleReport` outputs.
The [verified kernel and benchmark](examples/README.md#verified-kda-kernel)
are included as a reproducible example.

| Phase | Fixtures | Default deadline | Override (seconds) |
|---|---|---:|---|
| Torch golden on NPU, before design | Five fixtures covering both masks, tails and wide decay, compared with a CPU FP64 reference | 1800 s | `PYPTO_KDA_GOLDEN_TIMEOUT_S` |
| Kernel on NPU, after development | Same five fixtures, compared with FP64 and with the saved, validated golden outputs | 1800 s | `PYPTO_KDA_VERIFY_TIMEOUT_S` |
| Simulator smoke check per candidate | `(17, 7)` with the diagonal and `(2, 1)` with a wide gate step and strict lower mask | 900 s | `PYPTO_KDA_SMOKE_TIMEOUT_S` |
| Simulator score per candidate | `(128, 64)`, seed 0, diagonal included, cumulative gates | 1800 s | `PYPTO_KDA_SCORE_TIMEOUT_S` |
| Final selected kernel on NPU | Five correctness fixtures, then synchronized end-to-end latency on `(128, 64)` with three warmups and ten measured calls | 1800 s | `PYPTO_KDA_FINAL_TIMEOUT_S` |

The KDA entry point uses the KDA fixtures for every source path, including copies
of the Torch operator. For a different operator that uses its own standalone
self-check, explicitly set `PYPTO_COMBO_VERIFICATION=generic`.

When simulation selects a different kernel, the workflow verifies both the
starting kernel and the finalist on the NPU. Ten alternating timing pairs
decide deployment. The finalist needs nine wins and a median paired gain of
at least 1%. The final report records the selected source, simulator result,
hardware measurements and raw samples.

Both Python artifacts must compile before their fixtures run. On failure the
owning author receives the diagnostic and one retry. A second golden failure
ends the run before design. Both artifacts must return finite FP32 output
within `atol=rtol=1e-3` of FP64, including the retained diagonal of the
wide-step fixture. Masked entries must be exactly zero.

Each subprocess evaluation retains `stdout.log`, `stderr.log` and the simulator outputs in
its own directory under `evaluations/`. The score sums cycles across sequential
JIT components of the scoring fixture. Pipe and stall counters come from the
component with the highest cycle count.

The original `optimize` and `repair` commands use Claude ACP by default.
Agent calls use the signed-in connector account. KDA passes its own connector
and model settings to each nested stage.

```text
examples/           standalone kernels and the verified KDA implementation
fixtures/           deliberately broken kernels for repair and triage
torch_src/          source operators for the authoring workflows
src/                workflow entry points (see table above)
src/workflows/      reusable nested workflows
src/tasks/          deterministic workflow tasks
src/agents/         model-backed workflow actors
src/sources/        workflow inputs
src/viewers/        IDE result viewers
src/lib/            shared runner, records, config and report parsing
tools/              IR dumping and the KDA fixture driver
tests/              workflow regression tests
custom/             generated authoring and search workspaces
runs/               per-kernel reports, diffs and simulator output
wf-out/             wfpy journals, queue traces and agent transcripts
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

By default, `runs/` and `wf-out/` are inside the repository. wfpy defaults its `out_dir` to `./wf-out` relative to the
*invocation* cwd, which would otherwise scatter run logs; every workflow pins it
to `<repo>/wf-out` instead, overridable with `PYPTO_WF_OUT`. `./run.sh wfpy run
...` passes `--out-dir` for the same reason, since the bare `wfpy` CLI bypasses
the workflows' own `main()`.

## Run the agentic workflows

Each entry in the table is an IDE-openable `@workflow`. The four workflows below
default to Codex ACP and `gpt-6.2-sol` and accept the same `PYPTO_COMBO_MODEL`
and `PYPTO_COMBO_CONNECTOR` overrides as KDA. They require a working Ascend NPU
and the cannbot ops skills. To keep journals beside the IDE runs, launch from
the repository root with the following environment.

```bash
export PYPTO_WF_OUT="$PWD/src/wf-out"
export PYPTO_DEVICES=0
export TILE_FWK_DEVICE_ID="$PYPTO_DEVICES"
./run.sh npu-tune
./run.sh residual-rmsnorm
./run.sh precision-repair
```

**NPU-guided KDA tuning.** The verified example is the starting kernel. Each
round profiles the current kernel on the chosen NPU. The `tune-swimlane` agent
can request `tune-incore` for a measured task that persists in a fresh trace.
Proposed edits must pass two small simulator fixtures and five NPU fixtures.
The workflow alternates the incumbent and candidate in ten timing pairs with
profiling off. Adoption requires at least nine wins and a median paired gain of
at least 1%. The IDE shows the profile, trace, source diff and verdict. Set
`PYPTO_DEVICE_TUNE_BUDGET` to the maximum number of edits (default 3), or to
0 for a baseline verification and profile. A verified starting kernel remains eligible when every proposed edit is rejected. Workspaces are in `custom/device_tune/`.

**Competing residual RMSNorm designs.** A planner gives two designers the same
FP32 contract for `(normalized, residual_sum)`. One branch reuses the elementwise
residual sum across normalization and the other splits the feature reduction. Each gets its own
verification brief, implementation, one authoring retry, and the same NPU-tuning
budget. Before proposing edits, each tuner receives measured paired timings for its authored kernel and unfused PyTorch. The final comparison rotates the two candidates and unfused PyTorch on
one device, at 64 × 128 and a held-out 37 × 160. A candidate must win at least
nine of twelve rounds and improve median latency by at least 2% on **both**
shapes. Unfused PyTorch can win. Branch artifacts and the comparison report
are in `custom/residual_rmsnorm/`.

**Precision repair.** The bundled attention kernel normalizes along the wrong
axis. The workflow captures device checkpoints, asks the call-review and
binary-search agents to identify the first divergence, then gives a repair
agent the evidence. A scoped edit is accepted only after all three external
fixtures pass on the NPU. The frozen reference and fixture driver live outside
the editable candidate. `PYPTO_PRECISION_REPAIR_BUDGET` defaults to two attempts.
Check `custom/precision_repair/` for the device logs, diff and terminal verdict.

**Resumable PANKO search.** Install its tile-search dependency in the workflow
venv. Use a named workspace to continue the same campaign after an interruption.

```bash
.venv/bin/python -m pip install 'optuna>=2,<5'
export PYPTO_PANKO_WORKSPACE="$PWD/custom/panko/kda-demo"
export PYPTO_PANKO_BUDGET=6
./run.sh panko
```

The PANKO agent drives the installed vendor harness. For the tested CANN 9.2
build, the workspace keeps a versioned copy with adaptations for PyPTO enum
iteration, JSON replies from a reused worker and the nested JIT in the KDA seed.
The wrapper check permits edits inside that JIT body and freezes its host factory. Its tree contains
optimization intents. Each selected edit starts from the best measured program.
When the harness supplies a tile value, the graph skips the coder. Search checkpoints and worker state live in the campaign workspace. To resume, repeat the same
command with its original device, budget and fixture configuration. The
IDE shows the open intents and the final verdict. The optional
`PYPTO_PANKO_LINT_ROOT` points to the external hooks directory containing
`pypto_op_lint` when available. The setup uses the project KDA correctness
fixtures and harness feasibility checks on hosts without the hooks.

The default budget is six charged evaluations including the baseline. The
vendor objective is **AICore microseconds with profiling enabled**. Before
publishing a winner, the workflow checks the harness stop marker, source hash
and best measured node, disables the profiler in a delivery copy, and runs the
five fixtures and synchronized end-to-end timing on the NPU. If the budget
expires with a distinct, scored refinement pending, the workflow checks its
snapshot hash and correctness and compares it against the harness winner in
ten paired calls. The report names the recommended deployment kernel and retains
the vendor-delivered source. To compare with
another KDA run, set `PYPTO_PANKO_COMPARE_KERNEL` to that run's selected Python
file. The report includes ten paired end-to-end measurements against it. Use
the same seed kernel and record each search budget when comparing strategies.

For a preview of a workflow's nested graph, use the plan command.

```bash
./run.sh plan src/residual_rmsnorm.py --hierarchy
```

### Results from the November 25, 2026 NPU runs

The four workflows ran with Codex ACP and `gpt-6.2-sol` on separate Ascend
devices. With the output directory set as above, each workflow writes its
verdict to `src/wf-out/<run-id>/Out.md`.

| Command | Device | Hardware result |
|---|---:|---|
| `precision-repair` | 0 | The first repair changed the softmax axis from 0 to 1. All three fp16 fixtures passed (seeds 0, 17 and 29). |
| `npu-tune` | 1 | The three KDA edits won 0, 5 and 4 of their 10 timing pairs. The verified starting kernel remained the winner. |
| `residual-rmsnorm` | 2 | Both generated kernels passed. Unfused PyTorch had the lowest median latency at both 64 × 128 (0.289 ms) and 37 × 160 (0.269 ms). The faster generated design, row reuse, took 2.777 ms and 2.729 ms respectively. |
| `panko` | 3 | The six-evaluation search delivered the baseline at 582.58 µs of profiled AICore time. The last candidate reached 349.94 µs during refinement before the budget expired. |

A separate final check of that PANKO candidate passed all five NPU fixtures.
With profiling off at 128 × 64, its median synchronized call took 3.406 ms
and the delivered baseline took 3.542 ms. Each median came from a separate
ten-call run. The November 25 workflow delivered the baseline.

On November 26 the updated workflow replayed that budget stop on NPU 3.
Both snapshots passed all five fixtures. The pending candidate won all ten
alternating pairs with a median paired gain of 7.97% (median synchronized
calls of 3.744 ms and 3.491 ms), so the report recommended it for deployment.
The vendor search record still identifies its original winner.

The paired KDA check also replayed the earlier simulator reduction from
4,497,456 to 918,106 cycles. On NPU 6 the finalist won all ten timing pairs
with a 23.08% median paired gain (4.364 ms and 3.343 ms).

## Setup

PyPTO needs no build — CANN ships it. wfpy goes into the *same* venv so the
workflows and the kernels share one interpreter.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 -m venv --system-site-packages .venv
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
./.venv/bin/pip install torch_npu==2.9.0          # device runs; omit for sim only
./.venv/bin/pip install numpy pyyaml matplotlib pandas
./.venv/bin/pip install "wfpy[agent] @ git+https://github.com/huawei-csl/wfpy.git@6678b3309735fdd8dadad3c568a8d7825022aa24"
```

This is the wfpy 1.3.2 revision used to validate the KDA graph. Install and
authenticate its Codex connector on the host where the workflow runs.

```bash
npm install -g @openai/codex @agentclientprotocol/codex-acp
codex login
source .venv/bin/activate
wfpy connectors
```

The IDE extension needs `wfpy`, `wfpy-sidecar` and `codex-acp` on its executable
search path. Launch the IDE from the activated environment, or set
`wfpy.cliCommand` and `wfpy.sidecarCommand` in user settings or in the local
`.vscode/settings.json` to the absolute paths of the two executables in this
checkout’s `.venv/bin/`. Select `.venv/bin/python` as the Python interpreter.

For cannbot skills, the project looks in `../pypto/cannbot-skills/ops` relative
to the repository. Set `CANNBOT_OPS` to the collection's `ops` directory
if it lives elsewhere. An explicit `WF_SKILLS_ROOT` takes precedence.
Run the workflow regression suite with
`OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -q`.

> **Pin the torch/torch_npu pair; do not install "latest".** 2.9.0/2.9.0 works
> here (8 devices visible), as does 2.10.0/2.10.0. But `torch_npu` 2.13.0 and
> 2.14.0 each abort at import with a C++ `Duplicated key
> 'pinned_reserve_segment_size_mb'`, and because modern torch auto-loads the NPU
> backend through an entry point, a broken `torch_npu` breaks plain
> `import torch` and therefore *simulator* runs too.
>
> Use the CPU index for torch: the default index serves a `+cu130` build that
> drags ~3GB of unusable `nvidia-*` wheels onto an Ascend box.

`run.sh` sources the CANN env itself, so every command below works from a bare shell.

### Run mode: device or simulator

Workflows that use `default_run_mode()` select `npu` when the kernel interpreter
has `torch_npu` installed. Set `PYPTO_RUN_MODE=npu|sim` to override them.
The KDA fixture driver selects the mode per phase as shown in the table above.

- **`npu`** runs on the real hardware, and is verified: the original four
  numbered kernels passed on 910B2, spread over devices 0–3. Tensor mode fully supports this host — the
  Pro-mode arch limit below does *not* apply.
- **`sim`** is the Soc CAModel on CPU. No device needed, but its cycle counts are
  a **cost model, not a measurement**.

`./run.sh perf` always reads per-pipe counters from the simulator.
Use `./run.sh profile` for measured device timings. It enables
`runtime_debug_mode=1` and analyzes the device swimlane and per-core activity.

Concurrent slots get one device each via `TILE_FWK_DEVICE_ID`
(`PYPTO_DEVICES`, default `0,1,2,3,4,5,6,7`). Without that, pooled runs would
serialise on device 0 and the measured speed-up would be an artefact.

Standalone validation, simulation and IR jobs write to `runs/<kernel>_<mode-or-phase>/run-*/`.
The sibling `latest` symlink points to the most recently started invocation.
Standalone repair and optimization evaluate kernel edits with the initial
script’s inputs, reference and assertions. Their generated proposals and
checked sources are kept separately.

## Phase 1 — validate

Runs each kernel in the selected mode and checks its exit code and `PASS:` line.

```console
$ PYPTO_RUN_MODE=sim ./run.sh validate examples/0{1,2,3}_*.py
PASS  01_fused_elementwise   13.3s  (sim)
PASS  02_row_softmax         15.1s  (sim)
PASS  03_fused_linear        16.7s  (sim)
```

Failures route through the Gate's `Bad` port instead of crashing the run:

```console
$ ./run.sh validate fixtures/02_row_softmax_broken.py
```

### Concurrently, over a pool

`./run.sh pool` runs the kernels at the same time. One actor never runs
concurrently with *itself*, so real parallelism needs one worker chain per slot —
`PYPTO_POOL` sets how many (default 4). Workers take kernels in round-robin order, so one run can cover more kernels than slots. NPU concurrency is capped by the configured device count.

```console
$ ./run.sh pool examples/0{1,2,3}_*.py
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
      report: runs/03_fused_linear_ir/latest/ir_report.md
      diffs:  runs/03_fused_linear_ir/latest/ir_diffs
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
$ PYPTO_RUN_MODE=sim ./run.sh perf
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
> signal and the absolute cycle counts as indicative. Use `./run.sh profile` for device timings.

## Phase 4 — bounded repair loop

A loop in wfpy is an ordinary feedback edge with no backpressure, so it **must**
terminate by itself. `Budget` is the only exit: a guarded action repairs while
attempts remain, an unguarded fallback gives up.

The repair agent uses the Claude ACP connector by default. For one repair attempt
against the bundled broken softmax, install its adapter, sign in and run the fixture.

```bash
npm install -g @anthropic-ai/claude-code @agentclientprotocol/claude-agent-acp
claude
PYPTO_REPAIR_BUDGET=1 ./run.sh repair fixtures/02_row_softmax_broken.py
```

### cannbot skills

All 17 `cannbot-skills/ops/pypto-*` skills target Tensor mode, so they apply to
these kernels directly. The repair agent loads `pypto-precision-debug` (the right
one for a numeric mismatch); override with `PYPTO_REPAIR_SKILL`.

They are used **in place**, via `WF_SKILLS_ROOT`, which wfpy treats as its
highest-priority skill root. `run.sh` exports it for you.

Two wfpy limitations shaped this, both addressed in
[huawei-csl/wfpy#61](https://github.com/huawei-csl/wfpy/pull/61):

- A symlinked skill directory was skipped silently, because `_resolve_skill_dir()`
  called `.resolve()` before checking the result still sat under the root. That is
  how cannbot's `install-helper` installs every skill, so an installed collection
  was invisible to wfpy while being perfectly visible to Claude Code.
- `WF_SKILLS_ROOT` held one directory, and a skill name may not contain `/`
  (`_validate_skill_name`), so a sub-skill such as
  `pypto-op-perf-tune/perf-analyzer` could not be named at all. It now takes an
  `os.pathsep` list, and `src/lib/optimize_config.py` adds the `pypto-op-perf-tune`
  directory as a second root so `perf-analyzer` resolves — probing for the
  capability first, because on an older wfpy a two-path value would make the whole
  variable one bogus directory and break the repair loop as a side effect.

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
  reach a statically-sized pool as one batch per slot, so
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
