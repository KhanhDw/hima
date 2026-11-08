from __future__ import annotations

import torch

from .lif_soma import LIFSoma


class AdaptiveLIFSoma(LIFSoma):
    """LIF with a subthreshold + spike-triggered adaptation current ``w``.

        w_{t+1} = w_t + (a * V_t - w_t) / tau_w   (+ b on spike)
        V_{t+1} = leak * V_t + I_t - w_t

    Both updates read the state at time t (simultaneous Euler), so the result
    does not depend on update order. ``w`` keeps relaxing during the refractory
    period (only the membrane is held), as it does in the biological model.
    Requires ``tau_w >= 1`` (explicit Euler is unstable below that).
    """

    _state_names = LIFSoma._state_names + ("w",)

    def __init__(
        self,
        theta=1.0,
        leak=0.9,
        a=0.01,
        b=0.2,
        tau_w=10.0,
        refractory_period=2,
        surrogate_slope=10.0,
        surrogate=True,
        device=None,
        dtype=torch.float32,
        v_reset=0.0,
    ):
        if tau_w < 1.0:
            raise ValueError("tau_w must be >= 1 (time steps) for a stable update")
        self.a, self.b, self.tau_w = float(a), float(b), float(tau_w)
        super().__init__(
            theta=theta, leak=leak, refractory_period=refractory_period,
            surrogate_slope=surrogate_slope, surrogate=surrogate,
            device=device, dtype=dtype, v_reset=v_reset,
        )

    def _default_states(self, like, voltage=None):
        states = super()._default_states(like, voltage)
        states["w"] = torch.zeros_like(like)
        return states

    def step(self, current):
        current = self._prepare_input(current)
        self._ensure_state(current)
        refractory = self._refractory_mask()
        V, w = self.V, self.w

        inv_tau = 1.0 / self.tau_w
        w_next = torch.add(w * (1.0 - inv_tau), V, alpha=self.a * inv_tau)

        voltage = torch.add(current, V, alpha=self.leak) - w
        if refractory is not None:
            voltage = torch.where(refractory, V, voltage)

        spike = self._fire(voltage, self.theta, refractory)
        fired = spike.detach() != 0
        self.V = voltage.masked_fill(fired, self.v_reset)
        self.w = torch.add(w_next, spike, alpha=self.b)  # spike keeps surrogate grad
        self.update_refractory(fired)
        return spike