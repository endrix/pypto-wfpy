"""Unfused residual addition and RMS normalization used as the operator reference."""

import torch


def residual_rmsnorm(x, residual, weight, eps=1e-6):
    """Return the scaled row normalization and the unnormalized residual sum."""
    residual_sum = x + residual
    mean_square = residual_sum.square().mean(dim=-1, keepdim=True)
    normalized = residual_sum * torch.rsqrt(mean_square + eps) * weight
    return normalized, residual_sum
