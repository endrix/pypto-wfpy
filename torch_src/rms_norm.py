#!/usr/bin/env python3
# coding: utf-8
"""RMSNorm in PyTorch — the input to the authoring workflow.

This is the *source* for `./run.sh author`, not a PyPTO kernel. It is written as
an ordinary torch module, the way the thing being ported would actually arrive.

Chosen deliberately:

- **One reduction and a broadcast**, so it is not trivial — a mean of squares
  over the last axis, a reciprocal square root, then a scale. That exercises the
  axis question (`dim=-1`, not `dim=0`) that broke two of this project's
  fixtures, and the `pypto.mean` absence that the API stage has to notice.
- **Small enough to verify by eye**, so a wrong answer from the pipeline is
  recognisable as wrong rather than merely unexpected.
- **Not already in `examples/`**, so the authoring run cannot succeed by
  recognising a kernel it has seen. LayerNorm is there; RMSNorm is not, and the
  difference matters: RMSNorm does not subtract the mean, so an agent that
  pattern-matches to LayerNorm produces something that fails its own check.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation over the last axis.

    out = x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight

    Note what is NOT here: no mean subtraction and no bias. That is the whole
    difference from LayerNorm, and it is easy to add by mistake.
    """

    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps) * self.weight


def reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The same thing as a function, for a kernel to be checked against."""
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_square + eps) * weight


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, hidden = 64, 128
    x = torch.randn((batch, hidden), dtype=torch.float32)
    weight = torch.randn(hidden, dtype=torch.float32)

    module = RMSNorm(hidden)
    with torch.no_grad():
        module.weight.copy_(weight)
    torch.testing.assert_close(module(x), reference(x, weight), atol=1e-5, rtol=1e-5)

    # Every row of a plain RMSNorm has unit mean square. If this holds and the
    # elementwise check above also holds, the reduction is over the right axis.
    plain = reference(x, torch.ones(hidden))
    torch.testing.assert_close(
        plain.pow(2).mean(dim=-1), torch.ones(batch), atol=1e-4, rtol=1e-4
    )
    print(f"PASS: rms_norm ({batch}, {hidden})")
