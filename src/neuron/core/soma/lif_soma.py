
# lif_soma.py

from __future__ import annotations

from typing import Optional, Union

import torch

from .base_soma import BaseSoma, TensorLike
from .surrogate import _SurrogateSpike  # noqa: F401


class LIFSoma(BaseSoma):
    """
    Vectorized Leaky Integrate-and-Fire soma.

    Membrane equation:
        V_candidate = leak * V + I

    Refractory behavior:
        - Membrane potential is held during refractory.
        - Input current is discarded during refractory.
        - Refractory counter decreases once per simulation step.

    Spike:
        spike = surrogate(V_candidate - theta)

    Reset:
        V <- v_reset when spike occurs.

    Gradient:
        - Spike output retains surrogate gradient.
        - Reset decision is detached.
        - Membrane retains BPTT path between spikes.
    """

    def __init__(
        self,
        *,
        theta: float = 1.0,
        leak: float = 0.9,
        refractory_period: int = 2,
        surrogate_slope: float = 10.0,
        surrogate: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        v_reset: float = 0.0,
    ) -> None:

        super().__init__(
            refractory_period=refractory_period,
            surrogate_slope=surrogate_slope,
            surrogate=surrogate,
            v_init=v_reset,
            device=device,
            dtype=dtype,
        )

        if theta <= 0:
            raise ValueError("theta must be > 0.")

        if not 0.0 <= leak <= 1.0:
            raise ValueError("leak must be in [0, 1].")

        self.theta = float(theta)
        self.leak = float(leak)
        self.v_reset = float(v_reset)

        self.initialize_state()

    def step(
        self,
        input_current: torch.Tensor,
    ) -> torch.Tensor:

        refractory = self.is_refractory

        voltage = torch.add(
            input_current,
            self.V,
            alpha=self.leak,
        )

        voltage = torch.where(
            refractory,
            self.V,
            voltage,
        )

        spike = self._fire(
            voltage,
            self.theta,
        )

        fired = spike.detach() > 0

        self.V = torch.where(
            fired,
            torch.full_like(voltage, self.v_reset),
            voltage,
        )

        self.update_refractory(fired)

        return spike
