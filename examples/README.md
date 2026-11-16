# PyPTO Tensor-mode examples (wfpy workflow building blocks)

Three small, self-contained PyPTO kernels, each verified against a pure-torch
reference. They are deliberately shaped as workflow nodes: one file, one kernel,
a `--run_mode {sim,npu}` flag, a `PASS:`/`FAIL:` line on stdout, and exit code
0/1 so a runner can gate on them.

| File | Kernel | Units exercised |
|---|---|---|
| `01_fused_elementwise.py` | `relu(a * b + c)` | vector only |
| `02_row_softmax.py` | `softmax(x, dim=-1)` | vector + reduction (tile spans the full row) |
| `03_fused_linear.py` | `relu(a @ b + bias)` | cube matmul + vector tail |

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

Docker is not needed for device execution. If these pins ever stop resolving,
three venvs on this machine already run the working pair and are the reference:
`pto-kernels/.venv`, `pto-mlir-cannbench/.venv` (2.9.0) and
`workflow/old_wfpy/.venv` (2.10.0).

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

All four pass in both modes.

`--run_mode npu` runs on the real hardware; it reads `TILE_FWK_DEVICE_ID` to pick
a device (the box has 8x 910B2). A device run writes compiled kernels but no
performance data, so the `perf` workflow only works in `sim`.

## Skills

The `cannbot-skills/ops/pypto-*` skills (17 of them) all target Tensor mode, so
they apply to these examples directly. The orchestration chain they describe is
plan -> golden -> design -> verify-scaffold -> develop -> E2E verify -> perf-tune,
documented in `cannbot-skills/ops/pypto-orchestration-manual/references/agents.md`.
None of them cover `pypto_pro`; two explicitly exclude it.
