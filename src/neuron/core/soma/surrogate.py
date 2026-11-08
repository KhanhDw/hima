from __future__ import annotations

import torch


class SurrogateSpike(torch.autograd.Function):
    """Hard threshold forward, fast-sigmoid surrogate derivative backward.

    d spike / d x  ~=  1 / (1 + slope * |x|)^2
    """

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = float(slope)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):  # pyright: ignore[reportIncompatibleMethodOverride]
        (x,) = ctx.saved_tensors
        return grad_output / (1.0 + ctx.slope * x.abs()).square(), None


# Backward-compatible private name used by the old modules.
_SurrogateSpike = SurrogateSpike


def spike_fn(x: torch.Tensor, slope: float, surrogate: bool = True) -> torch.Tensor:
    """Heaviside(x). Uses the custom autograd Function only when a gradient
    can actually flow, so inference / no_grad pays for a single compare."""
    if surrogate and x.requires_grad and torch.is_grad_enabled():
        return SurrogateSpike.apply(x, slope)
    return (x >= 0).to(x.dtype)