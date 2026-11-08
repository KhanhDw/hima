from __future__ import annotations

import torch

from .base_soma import BaseSoma


class ELIFSoma(BaseSoma):
    """Exponential integrate-and-fire soma (forward Euler).

        C dV/dt = -g_L (V - E_L) + g_L * Delta_T * exp((V - V_T)/Delta_T) + I
        spike when V >= V_peak, then V <- V_reset (default E_L).

    ``dt`` is in the same time unit as C/g_L; ``refractory_period`` is counted
    in simulation steps.
    """

    def __init__(
        self,
        E_L=-65.0,
        V_T=-55.0,
        Delta_T=2.0,
        g_L=0.1,
        V_peak=0.0,
        dt=0.1,
        refractory_period=2,
        surrogate_slope=10.0,
        surrogate=True,
        device=None,
        dtype=torch.float32,
        C_m=1.0,
        V_reset=None,
    ):
        super().__init__(
            refractory_period, device=device, dtype=dtype,
            surrogate_slope=surrogate_slope, surrogate=surrogate, v_init=E_L,
        )
        if Delta_T <= 0 or g_L < 0 or dt <= 0 or C_m <= 0:
            raise ValueError("Delta_T, dt, C_m must be > 0; g_L must be >= 0")
        if V_peak <= V_T:
            raise ValueError("V_peak must be > V_T")
        if dt * g_L / C_m > 1.0:
            raise ValueError("dt * g_L / C_m must be <= 1 for a stable Euler step")
        self.E_L, self.V_T, self.Delta_T = float(E_L), float(V_T), float(Delta_T)
        self.g_L, self.V_peak, self.dt = float(g_L), float(V_peak), float(dt)
        self.C_m = float(C_m)
        self.V_reset = float(E_L if V_reset is None else V_reset)
        self.initialize_state()

    def step(self, current):
        current = self._prepare_input(current)
        self._ensure_state(current)
        refractory = self._refractory_mask()
        V = self.V

        k = self.dt / self.C_m
        # Exponent clamped for numerical safety; anything above V_peak is
        # discarded by the reset anyway.
        exp_term = torch.exp(((V - self.V_T) * (1.0 / self.Delta_T)).clamp_(max=20.0))
        candidate = V * (1.0 - k * self.g_L) + k * self.g_L * self.E_L
        candidate = candidate.add(exp_term, alpha=k * self.g_L * self.Delta_T)
        candidate = candidate.add(current, alpha=k)
        voltage = candidate if refractory is None else torch.where(refractory, V, candidate)

        spike = self._fire(voltage, self.V_peak, refractory)
        fired = spike.detach() != 0
        self.V = voltage.masked_fill(fired, self.V_reset)
        self.update_refractory(fired)
        return spike