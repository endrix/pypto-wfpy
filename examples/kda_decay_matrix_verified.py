#!/usr/bin/env python3
"""Compute the causal KDA decay matrix in PyPTO Tensor mode.

The kernel subtracts gates before exponentiating, so inclusive self-pairs
retain their dot product even for the wide-step gates [0, -1000].
"""

import argparse
from functools import lru_cache
import os

import pypto
import torch
from torch.nn.functional import pad


POSITION_BLOCK = 32  # F-2 increases position work per loop visit.
FEATURE_BLOCK = 64


# Layer G. Prepare physically aligned ND buffers and a local diagonal mask.
def _prepare_inputs(a, k, g, diagonal):
    rows, features = a.shape
    row_tail = (-rows) % POSITION_BLOCK
    feature_tail = (-features) % FEATURE_BLOCK
    if row_tail or feature_tail:
        padding = (0, feature_tail, 0, row_tail)
        a = pad(a, padding)
        k = pad(k, padding)
        if row_tail:
            g = torch.cat((g, g[-1:].expand(row_tail, -1)), dim=0)
        if feature_tail:
            g = pad(g, (0, feature_tail))

    positions = torch.arange(POSITION_BLOCK, dtype=torch.int32, device=a.device)
    keep = (positions[None, :] - positions[:, None] < diagonal).to(torch.float32)
    return a, k, g, keep


# Layer H. Load, subtract gates, select, and reduce one feature tile.
def pypto_load(a, k, g, row, col, feature):
    pypto.set_vec_tile_shapes(8, 64)
    shape = [POSITION_BLOCK, FEATURE_BLOCK]
    a_row = pypto.view(a, shape, [row, feature], valid_shape=shape)
    k_col = pypto.view(k, shape, [col, feature], valid_shape=shape)
    g_row = pypto.view(g, shape, [row, feature], valid_shape=shape)
    g_col = pypto.view(g, shape, [col, feature], valid_shape=shape)
    return a_row, k_col, g_row, g_col


def pypto_gate_difference(g_row, g_col):
    pypto.set_vec_tile_shapes(2, 8, 64)
    shape = [POSITION_BLOCK, POSITION_BLOCK, FEATURE_BLOCK]
    right = pypto.expand_clone(pypto.unsqueeze(g_col, 0), shape, valid_shape=shape)
    return pypto.sub(pypto.unsqueeze(g_row, 1), right)


def pypto_mask_exponent(diff, keep):
    pypto.set_vec_tile_shapes(2, 8, 64)
    # Expand before comparison so a materialized predicate has an aligned
    # feature stride across vector subgraphs.
    shape = [POSITION_BLOCK, POSITION_BLOCK, FEATURE_BLOCK]
    keep_wide = pypto.expand_clone(pypto.unsqueeze(keep, -1), shape, valid_shape=shape)
    return pypto.where(pypto.gt(keep_wide, 0.0), diff, 0.0)


def pypto_reduce_features(a_row, k_col, diff):
    pypto.set_vec_tile_shapes(2, 8, 64)
    weighted = pypto.mul(pypto.unsqueeze(a_row, 1), pypto.exp(diff))
    terms = pypto.mul(weighted, pypto.unsqueeze(k_col, 0))
    # Keep the column axis innermost so reduction stores have an aligned width.
    by_feature = pypto.transpose(terms, 1, 2)
    # F-17 sets the reduction tile in the transposed axis order.
    pypto.set_vec_tile_shapes(2, 64, 8)
    return pypto.sum(by_feature, dim=1, keepdim=False)


def pypto_store_zero(out, row, col):
    pypto.set_vec_tile_shapes(2, 8)
    zero = pypto.full([POSITION_BLOCK, POSITION_BLOCK], 0.0, pypto.DT_FP32)
    pypto.assemble(zero, [row, col], out)


def pypto_store_diagonal(total, keep, out, row, col):
    pypto.set_vec_tile_shapes(2, 8)
    tile = pypto.where(pypto.gt(keep, 0.0), total, 0.0)
    pypto.assemble(tile, [row, col], out)


# Layers I and J. Keep loop predicates directly in the Tensor JIT entry.
@lru_cache(maxsize=2)
def _kernel_for_mode(mode):
    @pypto.frontend.jit(new_ir=False, runtime_options={"run_mode": mode})
    def kda_decay_matrix_codex_kernel(
        a: pypto.Tensor[[pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32],
        k: pypto.Tensor[[pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32],
        g: pypto.Tensor[[pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32],
        keep: pypto.Tensor[[POSITION_BLOCK, POSITION_BLOCK], pypto.DT_FP32],
        out: pypto.Tensor[[pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32],
    ):
        for cb in pypto.loop(a.shape[0] // POSITION_BLOCK, unroll_list=[1]):
            for ib in pypto.loop(k.shape[0] // POSITION_BLOCK, unroll_list=[1]):
                row = cb * POSITION_BLOCK
                col = ib * POSITION_BLOCK
                if ib > cb:
                    pypto_store_zero(out, row, col)
                else:
                    acc = pypto.tensor([POSITION_BLOCK, POSITION_BLOCK], pypto.DT_FP32)
                    for fb in pypto.loop(
                        a.shape[1] // FEATURE_BLOCK, unroll_list=[1], submit_before_loop=True
                    ):
                        feature = fb * FEATURE_BLOCK
                        a_row, k_col, g_row, g_col = pypto_load(a, k, g, row, col, feature)
                        diff = pypto_gate_difference(g_row, g_col)
                        if ib == cb:
                            diff = pypto_mask_exponent(diff, keep)
                        partial = pypto_reduce_features(a_row, k_col, diff)

                        pypto.set_vec_tile_shapes(2, 8)
                        if pypto.is_loop_begin(fb):
                            total = partial
                        else:
                            total = pypto.add(acc, partial)
                        if pypto.is_loop_end(fb):
                            if ib == cb:
                                pypto_store_diagonal(total, keep, out, row, col)
                            else:
                                pypto.assemble(total, [row, col], out)
                        else:
                            acc[:] = total

    return kda_decay_matrix_codex_kernel


kda_decay_matrix_codex_kernel_npu = _kernel_for_mode(pypto.RunMode.NPU)


# Layer K. Bind contiguous ND data, allocate output, and launch once.
def kda_decay_matrix_codex_wrapper(a, k, g, diagonal=1, run_mode="npu"):
    """Return the FP32 decay matrix on the input device."""
    if a.ndim != 2 or k.shape != a.shape or g.shape != a.shape:
        raise ValueError("a, k, and g must have the same rank-2 shape")
    rows, features = a.shape
    if rows <= 0 or features <= 0:
        raise ValueError("BT and K must be positive")
    if any(x.dtype != torch.float32 or x.device != a.device for x in (a, k, g)):
        raise ValueError("a, k, and g must be FP32 tensors on the same device")
    if any(not x.is_contiguous() for x in (a, k, g)):
        raise ValueError("a, k, and g must be contiguous ND tensors")
    if not isinstance(diagonal, int) or diagonal not in (0, 1):
        raise ValueError("diagonal must be 0 or 1")

    if run_mode == "npu":
        kernel = kda_decay_matrix_codex_kernel_npu
    elif run_mode == "sim":
        kernel = _kernel_for_mode(pypto.RunMode.SIM)
    else:
        raise ValueError("run_mode must be 'npu' or 'sim'")

    a_pad, k_pad, g_pad, keep = _prepare_inputs(a, k, g, diagonal)
    padded_rows = a_pad.shape[0]
    out = torch.empty((padded_rows, padded_rows), dtype=torch.float32, device=a.device)
    kernel(a_pad, k_pad, g_pad, keep, out)
    if padded_rows != rows:
        return out[:rows, :rows].contiguous()
    return out


kda_run = kda_decay_matrix_codex_wrapper


def reference(a, k, g, diagonal=1):
    """Evaluate the direct FP64 oracle on CPU for the command-line check."""
    a, k, g = (x.to(device="cpu", dtype=torch.float64) for x in (a, k, g))
    masked = torch.ones((a.shape[0], a.shape[0]), dtype=torch.bool).triu(diagonal)
    diff = (g[:, None, :] - g[None, :, :]).masked_fill(masked[..., None], 0.0)
    terms = a[:, None, :] * diff.exp() * k[None, :, :]
    return terms.sum(-1).masked_fill(masked, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_mode", choices=("npu", "sim"), default="npu")
    args = parser.parse_args()
    try:
        device = torch.device("cpu")
        if args.run_mode == "npu":
            import torch_npu

            device = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}")
            torch_npu.npu.set_device(device)

        rng = torch.Generator().manual_seed(0)
        a = torch.randn((2, 1), generator=rng, dtype=torch.float32)
        k = torch.randn((2, 1), generator=rng, dtype=torch.float32)
        g = torch.tensor([[0.0], [-1000.0]], dtype=torch.float32)
        out = kda_run(a.to(device), k.to(device), g.to(device), 1, args.run_mode)
        if out.shape != (2, 2) or out.dtype != torch.float32 or out.device != device:
            raise AssertionError(f"unexpected output {out.shape}, {out.dtype}, {out.device}")
        actual = out.cpu()
        if not torch.isfinite(actual).all() or actual[0, 1].item() != 0.0:
            raise AssertionError("expected finite output and exact masked zero")
        torch.testing.assert_close(actual.double(), reference(a, k, g), atol=1e-3, rtol=1e-3)
    except Exception as exc:
        print(f"FAIL: kda_decay_matrix_codex mode={args.run_mode} {exc}")
        return 1
    print(f"PASS: kda_decay_matrix_codex mode={args.run_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
