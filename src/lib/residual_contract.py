"""Fixed contract shared by the residual RMSNorm authors and the independent verifier."""

from pathlib import Path

from src.lib.runner import REPO_ROOT

REFERENCE = REPO_ROOT / "torch_src/residual_rmsnorm.py"
TRAINING_SHAPES = ((8, 32), (17, 96), (64, 128))
TIMING_SHAPE = TRAINING_SHAPES[-1]
HELD_OUT_SHAPE = (37, 160)
EPSILON = 1e-6
ATOL = 1e-4
RTOL = 1e-4
BRANCHES = {
    "row_reuse": "Keep the row's residual sum available across the reduction and normalization. "
                 "Explain how the PyPTO graph reuses the intermediate.",
    "split_reduction": "Decompose the feature reduction into partial sums and combine them. "
                       "Explain where residual values are retained or recomputed.",
}


def verification_cases(mode):
    shapes = TRAINING_SHAPES[:1] if mode == "sim" else TRAINING_SHAPES
    cases = [dict(shape=shape, seed=i, cancel=False, eps=EPSILON) for i, shape in enumerate(shapes)]
    cases.append(dict(shape=shapes[0], seed=7, cancel=True, eps=EPSILON))
    if mode == "npu":
        cases.append(dict(shape=shapes[0], seed=3, cancel=False, eps=0.1))
    return cases


def contract_text() -> str:
    return (
        f"Frozen unfused PyTorch reference: {REFERENCE}\n\n"
        "Export residual_rmsnorm(x, residual, weight, eps=1e-6, run_mode='npu') from a standalone "
        "Python module. Compute with PyPTO Tensor mode in @pypto.frontend.jit-decorated code. "
        "Return (normalized, residual_sum), both FP32 and on the input device, with the same shape "
        "as x. Inputs x and residual are rank 2. Weight is a rank-1 vector over the last axis. "
        "Support variable row counts and feature widths aligned to 32. Preserve all inputs. "
        "Support CPU tensors in simulator mode and Ascend tensors in NPU mode. "
        "Use PyTorch only for allocating or adapting tensors at the kernel boundary.\n\n"
        f"Training shapes: {TRAINING_SHAPES}. Test random inputs, exact cancellation and a nondefault epsilon. "
        f"Compare both outputs to the CPU FP64 reference at atol={ATOL}, rtol={RTOL}. "
        f"Profile the {TIMING_SHAPE} shape. A held-out shape is tested after both branches finish."
    )


def implementation_brief(workspace: Path, through="VERIFICATION.md") -> str:
    parts = [contract_text()]
    for name in ("PLAN.md", "DESIGN.md", "VERIFICATION.md"):
        parts.extend((f"## {name}", (workspace / name).read_text(encoding="utf-8")))
        if name == through:
            break
    return "\n\n".join(parts)
