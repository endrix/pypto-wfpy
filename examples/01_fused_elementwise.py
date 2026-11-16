#!/usr/bin/env python3
# coding: utf-8
"""Fused elementwise chain on PyPTO (Tensor mode).

Computes ``out = relu(a * b + c)`` in a single PyPTO kernel, so the whole chain
stays resident in vector memory instead of round-tripping to HBM per op.

Run:
    python3 01_fused_elementwise.py --run_mode sim    # CPU Soc CAModel, no NPU needed
    python3 01_fused_elementwise.py --run_mode npu    # needs torch_npu + a device

Exit code is 0 on success, 1 on numeric mismatch, so a workflow runner can gate on it.
"""

import argparse
import os
import sys

import torch

import pypto

SHAPE = (64, 128)
# Vector tile: rows x cols of the elementwise work unit.
VEC_TILE = (32, 64)

# The jit decorator captures this dict by reference, so main() can switch the
# run mode after the kernel below has already been decorated.
runtime_options = {"run_mode": pypto.RunMode.NPU}


@pypto.frontend.jit(runtime_options=runtime_options)
def fused_elementwise_kernel(
    a: pypto.Tensor[[...], pypto.DT_FP32],
    b: pypto.Tensor[[...], pypto.DT_FP32],
    c: pypto.Tensor[[...], pypto.DT_FP32],
    out: pypto.Tensor[[...], pypto.DT_FP32],
):
    """out = relu(a * b + c), fused into one kernel."""
    pypto.set_vec_tile_shapes(*VEC_TILE)
    out[:] = pypto.relu(a * b + c)


def reference(a, b, c):
    """Pure-torch reference the kernel is checked against."""
    return torch.relu(a * b + c)


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
    a = torch.randn(SHAPE, dtype=torch.float32, device=device)
    b = torch.randn(SHAPE, dtype=torch.float32, device=device)
    c = torch.randn(SHAPE, dtype=torch.float32, device=device)
    out = torch.zeros(SHAPE, dtype=torch.float32, device=device)

    fused_elementwise_kernel(a, b, c, out)

    expected = reference(a, b, c)
    try:
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
    except AssertionError as exc:
        print(f"FAIL: fused_elementwise mismatch\n{exc}")
        return 1

    print(f"PASS: fused_elementwise {SHAPE} vec_tile={VEC_TILE} mode={args.run_mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
