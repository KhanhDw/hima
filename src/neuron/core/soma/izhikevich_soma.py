from __future__ import annotations

import torch

from .base_soma import BaseSoma


class IzhikevichSoma(BaseSoma):
    """Izhikevich neuron.

        v' = 0.04 v^2 + 5 v + 140 - u + I      (two half-steps, as in the paper)
        u' = a (b v - u)
        on v >= threshold:  v <- c,  u <- u + d

    With ``dt=1`` this reproduces the reference algorithm exactly. If a
    refractory period is used, V is held at its reset value while the recovery
    variable ``u`` keeps relaxing toward ``b * V``.
    """

    _state_names = BaseSoma._state_names + ("u",)

    def __init__(
        self,
        a=0.02,
        b=0.2,
        c=-65.0,
        d=8.0,
        dt=1.0,
        threshold=30.0,
        refractory_period=0,
        surrogate_slope=10.0,
        surrogate=True,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__(
            refractory_period, device=device, dtype=dtype,
            surrogate_slope=surrogate_slope, surrogate=surrogate, v_init=c,
        )
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if dt * a >= 1.0:
            raise ValueError("dt * a must be < 1 for a stable recovery update")
        self.a, self.b, self.c, self.d = float(a), float(b), float(c), float(d)
        self.dt = float(dt)
        self.threshold = float(threshold)
        self.initialize_state()

    def _default_states(self, like, voltage=None):
        states = super()._default_states(like, voltage)
        states["u"] = self.b * states["V"]
        return states

    def step(self, current):
        current = self._prepare_input(current)
        self._ensure_state(current)
        refractory = self._refractory_mask()
        V, u = self.V, self.u

        h = 0.5 * self.dt
        drive = current - u + 140.0  # constant over the two half-steps
        v1 = V + h * (0.04 * V * V + 5.0 * V + drive)
        v2 = v1 + h * (0.04 * v1 * v1 + 5.0 * v1 + drive)
        voltage = v2 if refractory is None else torch.where(refractory, V, v2)

        da = self.dt * self.a
        u_next = torch.add(u * (1.0 - da), voltage, alpha=da * self.b)

        spike = self._fire(voltage, self.threshold, refractory)
        fired = spike.detach() != 0
        self.V = voltage.masked_fill(fired, self.c)
        self.u = torch.add(u_next, spike, alpha=self.d)
        self.update_refractory(fired)
        return spike