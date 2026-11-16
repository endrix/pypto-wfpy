#!/usr/bin/env python3
# coding: utf-8
"""Row-wise softmax on PyPTO (Tensor mode).

Exercises a *reduction* kernel: unlike a pure elementwise chain, each output
element depends on a whole row.

The tile below spans the full reduction axis (``VEC_TILE[1] == SHAPE[1]``), which
is a reasonable default, but note that is a **performance** choice and not a
correctness one: PyPTO splits a reduction across tiles correctly, and this kernel
still passes with tiles as narrow as ``(8, 16)`` on a 128-wide reduction
(measured). The failure mode that does matter is reducing the wrong axis.

Run:
    python3 02_row_softmax.py --run_mode sim    # CPU Soc CAModel, no NPU needed
    python3 02_row_softmax.py --run_mode npu    # needs torch_npu + a device

Exit code is 0 on success, 1 on numeric mismatch.
"""

import argparse
import os
import sys

import torch

import pypto

SHAPE = (64, 128)
# Second dim spans the whole row so the reduction is complete within one tile.
VEC_TILE = (16, SHAPE[1])

runtime_options = {"run_mode": pypto.RunMode.NPU}


@pypto.frontend.jit(runtime_options=runtime_options)
def row_softmax_kernel(
    x: pypto.Tensor[[...], pypto.DT_FP32],
    out: pypto.Tensor[[...], pypto.DT_FP32],
):
    """out = softmax(x, dim=-1)."""
    pypto.set_vec_tile_shapes(*VEC_TILE)
    out[:] = pypto.softmax(x, dim=-1)


def reference(x):
    """Pure-torch reference the kernel is checked against."""
    return torch.softmax(x, dim=-1)


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
    x = torch.randn(SHAPE, dtype=torch.float32, device=device)
    out = torch.zeros(SHAPE, dtype=torch.float32, device=device)

    row_softmax_kernel(x, out)

    expected = reference(x)
    try:
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
        # Each row must also sum to 1; catches a partial reduction that happens
        # to be numerically close but normalized over the wrong extent.
        row_sums = out.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3)
    except AssertionError as exc:
        print(f"FAIL: row_softmax mismatch\n{exc}")
        return 1

    print(f"PASS: row_softmax {SHAPE} vec_tile={VEC_TILE} mode={args.run_mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
