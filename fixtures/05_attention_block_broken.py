#!/usr/bin/env python3
# coding: utf-8
"""BROKEN on purpose: the attention softmax reduces the wrong axis.

One op wrong out of six, and nothing about it is loud: `dim=0` normalises down
the columns rather than across each score row, which is still a valid
distribution of exactly the right shape, finite everywhere, and it compiles
and runs. The error enters at op 2 and is then carried through three more
matmuls, which is what makes it worth localising rather than eyeballing.

The first attempt at this fixture dropped the 1/sqrt(DIM) scale instead. That
PASSED: the inputs here are scaled by 0.1, so the scores are small, the
softmax is near-uniform with or without the factor, and the difference fell
inside the kernel's own 2e-2 tolerance. A fixture that looks broken and
passes is worse than no fixture.

The original is examples/05_attention_block.py.

A transformer block on PyPTO (Tensor mode): attention, then a feed-forward.

The longest kernel here, and the reason it exists is the op *chain*. The others
have one to three compute ops, which is too few for two things this project
wants to demonstrate:

- A triage that bisects over the op sequence has nothing to bisect with three
  ops: the search is one step, and a log-n narrowing over n=3 is theatre.
- An optimizer has little to find on a kernel small enough to run well on one
  core. 04_layer_norm is 64x128 -- 8K elements -- and ran at 100% utilization on
  a single core of 72 with no stalls at all.

Six `pypto.*` compute calls, four of them matmuls of three different shapes:

    scores = Q @ Kt                 matmul   (SEQ, DIM) @ (DIM, SEQ) -> (SEQ, SEQ)
    probs  = softmax(scores * s)    softmax  over the last axis
    ctx    = probs @ V              matmul   (SEQ, SEQ) @ (SEQ, DIM) -> (SEQ, DIM)
    h      = ctx @ W1               matmul   (SEQ, DIM) @ (DIM, FF)  -> (SEQ, FF)
    a      = relu(h)                relu
    y      = a @ W2                 matmul   (SEQ, FF)  @ (FF, DIM)  -> (SEQ, DIM)
    out    = y + ctx                residual

Note there is no `from __future__ import annotations` here, and there must not
be: PEP 563 turns annotations into strings, and `pypto.frontend.jit` reads the
real annotation objects to tell a tensor parameter from a scalar one. With it,
every parameter looks like a plain string and the first tensor passed is
rejected with "Non-tensor parameter 'q' must not be a torch.Tensor".

`Kt` is passed in already transposed rather than transposing inside the kernel.
That keeps this example about the matmul chain: a transpose feeding a matmul is
its own optimization topic (catalogue item F-13) and its own set of constraints.

The cube tiles are set per matmul rather than once for the kernel, because the
four matmuls have different shapes -- that is catalogue item F-16, differentiated
cube tile shapes for a kernel with two or more matmuls. One tiling that suits
(SEQ, DIM) @ (DIM, SEQ) will not suit (SEQ, FF) @ (FF, DIM).
"""

import argparse
import math
import os

import torch

import pypto

SEQ, DIM, FF = 128, 64, 128
SCALE = 1.0 / math.sqrt(DIM)

# Cube tiles per matmul: [L0A m,k] [L0B k,n] [L0C m,n].
QK_TILES = ([64, 64], [64, 64], [64, 64])     # (SEQ,DIM) @ (DIM,SEQ)
AV_TILES = ([64, 64], [64, 64], [64, 64])     # (SEQ,SEQ) @ (SEQ,DIM)
UP_TILES = ([64, 64], [64, 64], [64, 64])     # (SEQ,DIM) @ (DIM,FF)
DOWN_TILES = ([64, 64], [64, 64], [64, 64])   # (SEQ,FF)  @ (FF,DIM)

VEC_TILE_SEQ = (64, SEQ)   # spans a whole score row, so the softmax reduces in one tile
VEC_TILE_FF = (64, FF)     # the feed-forward activation
VEC_TILE_DIM = (64, DIM)   # the residual add

runtime_options = {"run_mode": pypto.RunMode.NPU}


@pypto.frontend.jit(runtime_options=runtime_options)
def attention_block_kernel(
    q: pypto.Tensor[[...], pypto.DT_FP16],
    kt: pypto.Tensor[[...], pypto.DT_FP16],
    v: pypto.Tensor[[...], pypto.DT_FP16],
    w1: pypto.Tensor[[...], pypto.DT_FP16],
    w2: pypto.Tensor[[...], pypto.DT_FP16],
    out: pypto.Tensor[[...], pypto.DT_FP16],
):
    """out = (relu(ctx @ W1) @ W2) + ctx, where ctx = softmax(Q @ Kt * s) @ V."""
    pypto.set_cube_tile_shapes(*QK_TILES)
    scores = pypto.matmul(q, kt, q.dtype)

    pypto.set_vec_tile_shapes(*VEC_TILE_SEQ)
    # BUG: the softmax normalises down the columns instead of across each
    # score row. dim=0 and dim=-1 both give a valid distribution of the right
    # shape, so nothing downstream complains.
    probs = pypto.softmax(scores * SCALE, dim=0)

    pypto.set_cube_tile_shapes(*AV_TILES)
    ctx = pypto.matmul(probs, v, q.dtype)

    pypto.set_cube_tile_shapes(*UP_TILES)
    hidden = pypto.matmul(ctx, w1, q.dtype)

    pypto.set_vec_tile_shapes(*VEC_TILE_FF)
    activated = pypto.relu(hidden)

    pypto.set_cube_tile_shapes(*DOWN_TILES)
    projected = pypto.matmul(activated, w2, q.dtype)

    pypto.set_vec_tile_shapes(*VEC_TILE_DIM)
    out[:] = projected + ctx


def reference(q, kt, v, w1, w2):
    """Pure-torch reference the kernel is checked against."""
    scores = torch.matmul(q, kt) * SCALE
    probs = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(probs, v)
    return torch.matmul(torch.relu(torch.matmul(ctx, w1)), w2) + ctx


def main() -> int:
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
    # Unit-scale inputs, deliberately. With small inputs the scores are tiny, the
    # softmax comes out near-uniform at 1/SEQ, and on a square score matrix a
    # softmax over the wrong axis then gives almost the same answer as the right
    # one -- which silently masks exactly the kind of bug this kernel exists to
    # expose. At unit scale Q @ Kt over DIM=64 lands near N(0, DIM), and the
    # 1/sqrt(DIM) factor brings it to roughly N(0, 1): a selective distribution.
    scale = 1.0
    q = torch.randn((SEQ, DIM), dtype=torch.float16, device=device) * scale
    kt = torch.randn((DIM, SEQ), dtype=torch.float16, device=device) * scale
    v = torch.randn((SEQ, DIM), dtype=torch.float16, device=device) * scale
    w1 = torch.randn((DIM, FF), dtype=torch.float16, device=device) * scale
    w2 = torch.randn((FF, DIM), dtype=torch.float16, device=device) * scale
    out = torch.zeros((SEQ, DIM), dtype=torch.float16, device=device)

    attention_block_kernel(q, kt, v, w1, w2, out)

    expected = reference(q, kt, v, w1, w2)
    try:
        # Loose for fp16 across four matmuls and a softmax; the reference runs in
        # fp16 too, so this compares like with like rather than against fp32.
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
    except AssertionError as exc:
        print(f"FAIL: attention_block mismatch\n{exc}")
        return 1

    print(f"PASS: attention_block ({SEQ}, {DIM}, ff={FF}) mode={args.run_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
