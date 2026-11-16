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
./.venv/bin/pip install numpy torch pyyaml matplotlib pandas
```

`import pypto` hard-depends on `torch` (via `pypto/cost_model.py`), so torch is
needed even for simulator runs.

**Device runs are not working from PyPI wheels here.** `torch_npu` 2.13.0 and
2.14.0, each paired with the matching upstream `torch`, both abort on import:

```
terminate called after throwing an instance of 'c10::ValueError'
  what():  Duplicated key 'pinned_reserve_segment_size_mb' found in
           device-specific configuration parser hook registration
```

It is a hard `terminate()`, not a catchable Python error. Modern torch
auto-loads the NPU backend through an entry point
(`torch/__init__.py:_import_device_backends`), and that registration collides;
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` plus an explicit `import torch_npu` fails the
same way. Note this breaks **simulator** runs too while `torch_npu` is
installed, because plain `import torch` triggers the autoload — so uninstall it
if you are not chasing device execution.

The supported route is Ascend's own torch build or the CANN Docker image, which
ships a known-good torch/torch_npu/CANN triple; `pro/docs/en/install/
prepare_environment.md` points at the image rather than pinning pip versions.

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

All three currently pass in sim mode.

`--run_mode npu` additionally needs `torch_npu` (not yet installed here) and
`TILE_FWK_DEVICE_ID` set; the box has 8x 910B2, which Tensor mode does support.

## Skills

The `cannbot-skills/ops/pypto-*` skills (17 of them) all target Tensor mode, so
they apply to these examples directly. The orchestration chain they describe is
plan -> golden -> design -> verify-scaffold -> develop -> E2E verify -> perf-tune,
documented in `cannbot-skills/ops/pypto-orchestration-manual/references/agents.md`.
None of them cover `pypto_pro`; two explicitly exclude it.
