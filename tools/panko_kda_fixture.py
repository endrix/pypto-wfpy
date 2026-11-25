"""Frozen PyTorch golden and measurement inputs for the PANKO KDA campaign."""

import torch


def build_inputs(device):
    from tools.check_kda import inputs

    return tuple(t.to(f"npu:{device}") for t in inputs(128, 64, "cumulative"))


def kda_golden(a, k, g, diagonal=1):
    # Mask the exponent before exp so even an excluded wide-step pair stays finite.
    a, k, g = a.double(), k.double(), g.double()
    mask = torch.ones((a.shape[0], a.shape[0]), device=a.device, dtype=torch.bool).triu(diagonal)
    decay = (g[:, None] - g[None, :]).masked_fill(mask[..., None], 0).exp()
    return (a[:, None] * decay * k[None, :]).sum(-1).masked_fill(mask, 0).float()
