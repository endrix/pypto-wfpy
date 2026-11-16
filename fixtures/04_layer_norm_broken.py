#!/usr/bin/env python3
# coding: utf-8
"""LayerNorm on PyPTO (Tensor mode).

The hardest of these kernels, and the most realistic: it chains two reductions
over the hidden axis (mean, then variance of the centred values), broadcasts each
``(N, 1)`` result back against ``(N, H)``, and finishes with a scale and shift.

Three things have to be right at once, and none of them fail loudly:

1. The vector tile must span the **whole** hidden axis, or each reduction is
   computed over part of a row and the output is plausible but wrong.
2. Both reductions must be over ``dim=-1``. Reducing ``dim=0`` also produces a
   correctly shaped, finite result.
3. ``pypto.mean`` does not exist in this version -- a mean is ``pypto.sum`` over
   the axis divided by the axis length.

Run:
    python3 04_layer_norm.py --run_mode sim
    python3 04_layer_norm.py --run_mode npu

Exit code is 0 on success, 1 on numeric mismatch.
"""

import argparse
import os
import sys

import torch

import pypto

BATCH, HIDDEN = 64, 128
# Second dim spans the whole hidden axis: both reductions run inside one tile.
VEC_TILE = (64, 64)
EPS = 1e-5

runtime_options = {"run_mode": pypto.RunMode.NPU}


@pypto.frontend.jit(runtime_options=runtime_options)
def layer_norm_kernel(
    x: pypto.Tensor[[...], pypto.DT_FP32],
    gamma: pypto.Tensor[[...], pypto.DT_FP32],
    beta: pypto.Tensor[[...], pypto.DT_FP32],
    out: pypto.Tensor[[...], pypto.DT_FP32],
):
    """out = (x - mean) / sqrt(var + eps) * gamma + beta, over the last axis."""
    pypto.set_vec_tile_shapes(*VEC_TILE)
    hidden = x.shape[1]

    # No pypto.mean in this version: sum over the axis, then divide by its length.
    mean = pypto.sum(x, dim=-1, keepdim=True) / hidden
    centered = x - mean
    var = pypto.sum(x * x, dim=-1, keepdim=True) / hidden

    normalized = centered / pypto.sqrt(var + EPS)
    out[:] = normalized * gamma + beta


def reference(x, gamma, beta):
    """Pure-torch reference the kernel is checked against."""
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=gamma, bias=beta, eps=EPS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_mode", choices=["npu", "sim"], default="sim")
    args = parser.parse_args()

    if args.run_mode == "sim":
        runtime_options["run_mode"] = pypto.RunMode.SIM
        device = "cpu"
    else:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            print("FAIL: torch_npu is not installed; use --run_mode sim")
            return 1
        device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        runtime_options["run_mode"] = pypto.RunMode.NPU
        device = f"npu:{device_id}"

    torch.manual_seed(0)
    x = torch.randn((BATCH, HIDDEN), dtype=torch.float32, device=device)
    gamma = torch.randn(HIDDEN, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN, dtype=torch.float32, device=device)
    out = torch.zeros((BATCH, HIDDEN), dtype=torch.float32, device=device)

    layer_norm_kernel(x, gamma, beta, out)

    expected = reference(x, gamma, beta)
    try:
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
        # Normalize with gamma=1, beta=0 and every row must have zero mean and
        # unit variance. A reduction over the wrong axis, or a tile that covers
        # only part of a row, stays finite and correctly shaped -- so the
        # elementwise check above can pass on a sliced input while this fails.
        ones = torch.ones(HIDDEN, dtype=torch.float32, device=device)
        zeros = torch.zeros(HIDDEN, dtype=torch.float32, device=device)
        plain = torch.zeros((BATCH, HIDDEN), dtype=torch.float32, device=device)
        layer_norm_kernel(x, ones, zeros, plain)
        row_mean = plain.mean(dim=-1)
        row_var = plain.var(dim=-1, unbiased=False)
        torch.testing.assert_close(row_mean, torch.zeros_like(row_mean), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(row_var, torch.ones_like(row_var), atol=1e-2, rtol=1e-2)
    except AssertionError as exc:
        print(f"FAIL: layer_norm mismatch\n{exc}")
        return 1

    print(f"PASS: layer_norm ({BATCH}, {HIDDEN}) vec_tile={VEC_TILE} mode={args.run_mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
