
from __future__ import annotations

from typing import Any

import torch


class SurrogateSpike(torch.autograd.Function):
    """
    Hard threshold forward, fast-sigmoid surrogate derivative backward.

    Forward:
        spike = 1 if x >= 0 else 0

    Backward:
        dS/dx = 1 / (1 + slope * |x|)^2
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        slope: float,
    ) -> torch.Tensor:

        if slope <= 0:
            raise ValueError("slope must be > 0.")

        ctx.save_for_backward(x)
        ctx.slope = float(slope)

        return (x >= 0).to(dtype=x.dtype)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:

        (x,) = ctx.saved_tensors
        (grad_out,) = grad_outputs

        grad_x = grad_out / (
            1.0 + ctx.slope * x.abs()
        ).square()

        return grad_x, None



# Backward compatibility
_SurrogateSpike = SurrogateSpike


def spike_fn(
    x: torch.Tensor,
    slope: float = 10.0,
    surrogate: bool = True,
) -> torch.Tensor:
    """
    Generate binary spikes with optional surrogate gradient.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")

    if slope <= 0:
        raise ValueError("slope must be > 0.")

    if surrogate and x.requires_grad and torch.is_grad_enabled():
        return SurrogateSpike.apply(x, float(slope))

    return (x >= 0).to(dtype=x.dtype)
