#!/usr/bin/env python3
# coding: utf-8
"""Fused linear layer on PyPTO (Tensor mode): relu(a @ b + bias).

This is the first example that uses *both* compute units in one kernel: the cube
unit for the matmul and the vector unit for the bias-add and activation. Note the
two different tile-shape calls -- `set_cube_tile_shapes` before the matmul and
`set_vec_tile_shapes` before the elementwise tail. That interleaving is the
pattern the repo's own FFN module uses.

`bias` is a full (M, N) tensor rather than a broadcast (1, N) row, to keep this
example about the cube/vector fusion rather than about broadcast semantics.

Run:
    python3 03_fused_linear.py --run_mode sim    # CPU Soc CAModel, no NPU needed
    python3 03_fused_linear.py --run_mode npu    # needs torch_npu + a device

Exit code is 0 on success, 1 on numeric mismatch.
"""

import argparse
import os
import sys

import torch

import pypto

M, K, N = 64, 128, 64
# Cube tiles: [L0A m,m] [L0B k,k] [L0C n,n] -- mirrors the repo's beginner matmul.
CUBE_TILE_A = [32, 32]
CUBE_TILE_B = [64, 64]
CUBE_TILE_C = [64, 64]
VEC_TILE = (32, 64)

runtime_options = {"run_mode": pypto.RunMode.NPU}


@pypto.frontend.jit(runtime_options=runtime_options)
def fused_linear_kernel(
    a: pypto.Tensor[[...], pypto.DT_FP16],
    b: pypto.Tensor[[...], pypto.DT_FP16],
    bias: pypto.Tensor[[...], pypto.DT_FP16],
    out: pypto.Tensor[[...], pypto.DT_FP16],
):
    """out = relu(a @ b + bias), cube matmul fused with a vector tail."""
    pypto.set_cube_tile_shapes(CUBE_TILE_A, CUBE_TILE_B, CUBE_TILE_C)
    acc = pypto.matmul(a, b, a.dtype)

    pypto.set_vec_tile_shapes(*VEC_TILE)
    out[:] = pypto.relu(acc + bias)


def reference(a, b, bias):
    """Pure-torch reference the kernel is checked against."""
    return torch.relu(torch.matmul(a, b) + bias)


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
    a = torch.randn(M, K, dtype=torch.float16, device=device)
    b = torch.randn(K, N, dtype=torch.float16, device=device)
    bias = torch.randn(M, N, dtype=torch.float16, device=device)
    out = torch.zeros((M, N), dtype=torch.float16, device=device)

    fused_linear_kernel(a, b, bias, out)

    expected = reference(a, b, bias)
    try:
        # fp16 matmul over K=128 accumulates error; tolerance is looser than the
        # fp32 elementwise examples on purpose.
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    except AssertionError as exc:
        print(f"FAIL: fused_linear mismatch\n{exc}")
        return 1

    print(f"PASS: fused_linear ({M}x{K})@({K}x{N}) mode={args.run_mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
