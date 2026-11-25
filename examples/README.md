# PyPTO Tensor-mode examples (wfpy workflow building blocks)

Each example runs as a standalone kernel with `--run_mode sim` or
`--run_mode npu`. Its exit code and `PASS:` or `FAIL:` line carry the result
back to the workflow.

| File | Kernel | Units exercised |
|---|---|---|
| `01_fused_elementwise.py` | `relu(a * b + c)` | vector only |
| `02_row_softmax.py` | `softmax(x, dim=-1)` | vector + reduction (tile spans the full row) |
| `03_fused_linear.py` | `relu(a @ b + bias)` | cube matmul + vector tail |
| `04_layer_norm.py` | Layer normalization | vector + row reductions |
| `05_attention_block.py` | Attention and feed-forward block | cube matmuls + vector |
| [kda_decay_matrix_verified.py](kda_decay_matrix_verified.py) | KDA decayed Gram matrix | tiled feature reduction + masking |

## Verified KDA kernel

The workflow selected `candidate2.py` in run
`2026-11-25T08-50-46-923Z_64bf5633`. The example is a byte-for-byte copy
of `custom/kda_decay_matrix_codex/76688438f86f43f2b739d47b6b6ab1b9/candidates/candidate2.py`.
Its SHA-256 is `86a898aabbb4d4963ed32467d04bf7856f289060db85f80c7153f71748b226a4`.

| Metric | Authored baseline | Selected kernel |
|---|---:|---:|
| Simulator cycles, 128 × 64 | 4,497,456 | 918,106 (79.6% lower) |
| Median synchronized NPU latency | 4.094 ms | 3.303 ms (19.3% lower) |

The selected kernel passed all five correctness fixtures on Ascend 910B2 device 0.
Its largest absolute error against the FP64 reference was `2.85765e-6`.
The timing used seed 0, the diagonal included, three warmups and ten timed
calls at 128 × 64. The baseline timing came from a separate device evaluation
with the same fixture and timing procedure.

To run the example’s standalone wide-decay self-check on an Ascend host:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
TILE_FWK_DEVICE_ID=0 .venv/bin/python examples/kda_decay_matrix_verified.py --run_mode npu
```

The [full KDA workflow](../README.md#kda-workflow) runs all five fixtures and times the selected kernel.

## Why Tensor mode and not Pro mode

PyPTO Pro (`pypto_pro`) needs both a full source build of 0.3.0 **and**
950PR/950DT silicon: `pypto_pro/runtime/platform.py` maps only `DAV_3510` and
`DAV_9201`. This machine reports `DAV_2201` (910B2), so Pro kernels cannot
execute here. Tensor mode runs today with zero build.

## Environment

PyPTO is already installed as part of CANN — no build required. CANN >= 9.1.0
bundles it (`pro/docs/en/install/build_and_install.md`). Only pip-level deps are
missing from a bare interpreter.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # puts CANN's site-packages on PYTHONPATH

python3 -m venv --system-site-packages .venv
./.venv/bin/pip install numpy torch==2.9.0 pyyaml matplotlib pandas
```

`import pypto` hard-depends on `torch` (via `pypto/cost_model.py`), so torch is
needed even for simulator runs.

**Device runs work.** The torch/torch_npu pair must be pinned:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
pip install torch_npu==2.9.0      # -> torch.npu.is_available() True, 8 devices
```

2.10.0/2.10.0 also works. But `torch_npu` 2.13.0 and 2.14.0, each with its
matching upstream torch, both abort on import:

```
terminate called after throwing an instance of 'c10::ValueError'
  what():  Duplicated key 'pinned_reserve_segment_size_mb' found in
           device-specific configuration parser hook registration
```

A hard `terminate()`, not a catchable Python error. Modern torch auto-loads the
NPU backend through an entry point (`torch/__init__.py:_import_device_backends`),
and that registration collides; `TORCH_DEVICE_BACKEND_AUTOLOAD=0` plus an
explicit `import torch_npu` fails the same way. Note a broken `torch_npu` breaks
**simulator** runs too, because plain `import torch` triggers the autoload.

Note the installed package is pypto **0.2.1** (from CANN 9.2.0-beta.2), while the
`pro/` source tree is 0.3.0. Do not put `pro/python` on `PYTHONPATH` to "upgrade"
it: the 0.3.0 Python calls into a newer pybind binding and fails on import with
`module 'pypto.pypto_impl' has no attribute 'ExternalError'`.

## Running

`sim` uses the Soc CAModel on CPU — no NPU needed, and it also emits a swimlane
trace and bubble-analysis report under `./output/`.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./.venv/bin/python examples/01_fused_elementwise.py --run_mode sim
```

`--run_mode npu` runs on the real hardware; it reads `TILE_FWK_DEVICE_ID` to pick
a device (the box has 8x 910B2). A device run writes compiled kernels but no
performance data, so the `perf` workflow only works in `sim`.

## Skills

The `cannbot-skills/ops/pypto-*` skills (17 of them) all target Tensor mode, so
they apply to these examples directly. The orchestration chain they describe is
plan -> golden -> design -> verify-scaffold -> develop -> E2E verify -> perf-tune,
documented in `cannbot-skills/ops/pypto-orchestration-manual/references/agents.md`.
None of them cover `pypto_pro`; two explicitly exclude it.
