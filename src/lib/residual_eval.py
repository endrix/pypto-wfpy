"""Frozen simulator checks, NPU checks and timing inputs for residual RMSNorm."""

from functools import partial

import torch

from src.lib.residual_contract import ATOL, EPSILON, RTOL, TIMING_SHAPE, verification_cases
from torch_src.residual_rmsnorm import residual_rmsnorm as unfused


def inputs(shape, seed, cancel=False):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=rng)
    residual = -x if cancel else torch.randn(shape, generator=rng)
    weight = torch.randn(shape[-1], generator=rng)
    return x, residual, weight


def check_case(implementation, shape, seed, mode, device, *, cancel=False, eps=EPSILON):
    cpu = inputs(shape, seed, cancel)
    expected = unfused(*(value.double() for value in cpu), eps=eps)
    args = [value.to(device, copy=True) for value in cpu]
    try:
        actual = implementation(*args, eps=eps, run_mode=mode)
    except Exception as exc:
        raise RuntimeError(f"{mode} shape={shape}, seed={seed}, eps={eps}, cancel={cancel}: {exc}") from exc
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise ValueError("expected (normalized, residual_sum)")
    for name, value, target in zip(("normalized", "residual_sum"), actual, expected):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device != device:
            raise ValueError(f"{name} must be an FP32 tensor on {device}")
        torch.testing.assert_close(value.detach().cpu().double(), target, rtol=RTOL, atol=ATOL,
                                   msg=lambda m: f"{name} on {shape}: {m}")
    for name, tensor, original in zip(("x", "residual", "weight"), args, cpu):
        torch.testing.assert_close(tensor.cpu(), original, rtol=0, atol=0,
                                   msg=lambda m: f"modified input {name} on {shape}: {m}")


def check(module, mode, device):
    for case in verification_cases(mode):
        check_case(module.residual_rmsnorm, mode=mode, device=device, **case)


def timed_call(module, shape, device):
    tensors = tuple(t.to(device) for t in inputs(shape, 27))
    if module is None:
        return partial(unfused, *tensors, eps=EPSILON)
    return partial(module.residual_rmsnorm, *tensors, eps=EPSILON, run_mode="npu")


def benchmark(module, device):
    """Bind input tensors before the shared tuning workflow starts its timer."""
    return timed_call(module, TIMING_SHAPE, device)
