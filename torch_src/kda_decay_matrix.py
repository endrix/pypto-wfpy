#!/usr/bin/env python3
# coding: utf-8
"""One stage of KDA in PyTorch — the decayed Gram matrix.

The input to `./run.sh author`, and then to `./run.sh optimize`. This is stage 1
and stage 4 of Kimi Delta Attention (chunked linear attention with per-dimension
log-space decay gates), lifted out of
`kda_097/artifacts/kda_vectorized.py` as a standalone kernel
with its own reference and self-check.

    A[c, i] = sum_d  a[c, d] * exp(g[c, d] - g[i, d]) * k[i, d]      for i <= c
            = 0                                                      otherwise

Why this stage and not the whole algorithm: the other five stages need a
recursive blocked triangular inverse and a scan with a loop-carried dependence
over chunks. pypto has `loop`, `concat` and `assemble`, so both are expressible,
but neither is a tile-shaped problem and they would dominate an authoring run.
This stage is exactly tile-shaped — two matmuls, an exponential and a
triangular mask — and it is where the arithmetic actually goes.

**The numerical trap, and the reason this is a fair test.** Written directly,
`exp(g[c,d]) * exp(-g[i,d])` overflows fp32: `g` is a cumulative sum of
log-space gates, so it grows with the chunk, and the two factors are enormous
and tiny respectively. The KDA report for this machine records that the naive
factorisation produces NaN at m=128, measured. The fix is to compute blockwise
against a per-row-block reference exponent `r`, so both factors stay in (0, 1]:

    lhs = a[c0:c0+m] * exp(g[c0:c0+m] - r)        <- r is the block's first row
    rhs = k[i0:i0+m] * exp(r - g[i0:i0+m])
    A[block] = lhs @ rhs^T

A port that ignores this passes a loose tolerance on small random inputs and
produces NaN on real ones, which is the kind of mistake worth being able to
catch.
"""

import torch

# Row-block size for the reference-exponent trick. m=128 overflows fp32;
# the KDA report for this machine measured NaN there.
M_BLK = 32


def decay_matrix(
    a: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    diagonal: int = 1,
    m: int = M_BLK,
) -> torch.Tensor:
    """Lower-triangular decayed Gram matrix, blocked for fp32 safety.

    a, k, g: (BT, K). Returns (BT, BT).
    `diagonal=0` masks the diagonal out (strict lower), `diagonal=1` keeps it.
    """
    bt, _ = a.shape
    out = a.new_zeros(bt, bt)
    for c0 in range(0, bt, m):
        r = g[c0:c0 + 1, :]                                  # the block's reference row
        lhs = a[c0:c0 + m, :] * (g[c0:c0 + m, :] - r).exp()  # both factors in (0, 1]
        for i0 in range(0, c0 + m, m):
            rhs = k[i0:i0 + m, :] * (r - g[i0:i0 + m, :]).exp()
            out[c0:c0 + m, i0:i0 + m] = lhs @ rhs.transpose(-1, -2)
    mask = torch.triu(torch.ones(bt, bt, dtype=torch.bool, device=a.device), diagonal)
    return out.masked_fill(mask, 0)


def reference(a: torch.Tensor, k: torch.Tensor, g: torch.Tensor, diagonal: int = 1):
    """The same thing, for a kernel to be checked against."""
    return decay_matrix(a, k, g, diagonal)


def _naive(a: torch.Tensor, k: torch.Tensor, g: torch.Tensor, diagonal: int = 1):
    """The unblocked factorisation, kept to show why the blocking exists.

    exp(g[c]) * exp(-g[i]) with no reference exponent. Correct on paper and
    unusable in fp32.
    """
    bt = a.shape[0]
    out = (a * g.exp()) @ (k * (-g).exp()).transpose(-1, -2)
    mask = torch.triu(torch.ones(bt, bt, dtype=torch.bool, device=a.device), diagonal)
    return out.masked_fill(mask, 0)


def main() -> int:
    torch.manual_seed(0)
    bt, kdim = 128, 64

    a = torch.randn((bt, kdim), dtype=torch.float32)
    k = torch.randn((bt, kdim), dtype=torch.float32)
    # g as KDA builds it: a cumulative sum of negative log-space gates, so it
    # decreases down the chunk and spans a wide range by the last row.
    # Scaled so the cumulative sum spans beyond fp32's exponent range by the
    # last row (|g| > ~88), which is what makes the naive factorisation overflow
    # and therefore what makes this fixture discriminate between a correct port
    # and one that skipped the blocking. With milder gates both agree and the
    # test proves nothing -- measured: at *0.5 the naive version stays finite.
    gates = -torch.rand((bt, kdim), dtype=torch.float32) * 1.5
    g = gates.cumsum(0)

    out = decay_matrix(a, k, g)

    # 1. The mask holds: strictly upper entries are zero.
    upper = torch.triu(torch.ones(bt, bt, dtype=torch.bool), 1)
    assert out.masked_select(upper).abs().max() == 0, "upper triangle is not zero"

    # 2. Finite everywhere -- the whole point of the blocking.
    assert torch.isfinite(out).all(), "blocked result is not finite"

    # 3. Against an fp64 oracle computed entry by entry, which is the definition.
    a64, k64, g64 = a.double(), k.double(), g.double()
    oracle = torch.zeros(bt, bt, dtype=torch.float64)
    for c in range(bt):
        for i in range(c + 1):
            oracle[c, i] = (a64[c] * (g64[c] - g64[i]).exp() * k64[i]).sum()
    torch.testing.assert_close(out.double(), oracle, atol=1e-3, rtol=1e-3)

    # 4. And the naive factorisation really does fail, so the blocking is not
    #    decoration. Reported rather than asserted: it overflows to inf/nan on
    #    this data, and saying so is the point.
    naive = _naive(a, k, g)
    if torch.isfinite(naive).all():
        print("       WARNING: the naive factorisation stayed finite, so this "
              "input does not discriminate a port that skipped the blocking")
    else:
        bad = (~torch.isfinite(naive)).sum().item()
        print(f"       naive (unblocked) factorisation: {bad} non-finite entries "
              f"-- the blocking is load-bearing")

    print(f"PASS: kda_decay_matrix ({bt}, {kdim}) m={M_BLK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
