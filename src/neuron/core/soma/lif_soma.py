from __future__ import annotations

import torch

from .base_soma import BaseSoma
from .surrogate import _SurrogateSpike  # noqa: F401  (re-exported for old imports)


class LIFSoma(BaseSoma):
    """Vectorized Leaky Integrate-and-Fire soma.

        V_{t+1} = leak * V_t + I_t           (held while refractory)
        spike   = V_{t+1} >= theta
        on spike: V <- v_reset, refractory counter <- refractory_period

    Input arriving during the refractory period is discarded. The reset is a
    discrete (detached) decision; the spike output keeps the surrogate
    gradient, and the membrane keeps a full BPTT path between spikes.
    """

    def __init__(
        self,
        theta=1.0,
        leak=0.9,
        refractory_period=2,
        surrogate_slope=10.0,
        surrogate=True,
        device=None,
        dtype=torch.float32,
        v_reset=0.0,
    ):
        super().__init__(
            refractory_period, device=device, dtype=dtype,
            surrogate_slope=surrogate_slope, surrogate=surrogate, v_init=v_reset,
        )
        if theta <= 0:
            raise ValueError("theta must be > 0")
        if not 0.0 <= leak <= 1.0:
            raise ValueError("leak must be in [0, 1]")
        self.theta = float(theta)
        self.leak = float(leak)
        self.v_reset = float(v_reset)
        self.initialize_state()

    def step(self, current):
        current = self._prepare_input(current)
        self._ensure_state(current)
        refractory = self._refractory_mask()

        voltage = torch.add(current, self.V, alpha=self.leak)  # leak*V + I
        if refractory is not None:
            voltage = torch.where(refractory, self.V, voltage)

        spike = self._fire(voltage, self.theta, refractory)
        fired = spike.detach() != 0
        self.V = voltage.masked_fill(fired, self.v_reset)
        self.update_refractory(fired)
        return spike