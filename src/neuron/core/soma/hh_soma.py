from __future__ import annotations

import torch

from .base_soma import BaseSoma
from .surrogate import spike_fn


def _exp(x):
    return torch.exp(x.clamp(-50.0, 50.0))


def _vtrap(x):
    """x / expm1(x) with the analytic limit 1 at x -> 0 (NaN-free gradients)."""
    x = x.clamp(-60.0, 60.0)
    small = x.abs() < 1e-6
    safe = x.masked_fill(small, 1.0)
    return torch.where(small, 1.0, safe / torch.expm1(safe))


def hh_rates(v):
    """Classic HH rate functions (rest at V = 0, depolarisation positive)."""
    am = _vtrap(2.5 - 0.1 * v)
    bm = 4.0 * _exp(v * (-1.0 / 18.0))
    ah = 0.07 * _exp(v * (-1.0 / 20.0))
    bh = torch.reciprocal(1.0 + _exp(3.0 - 0.1 * v))
    an = 0.1 * _vtrap(1.0 - 0.1 * v)
    bn = 0.125 * _exp(v * (-1.0 / 80.0))
    return am, bm, ah, bh, an, bn


def _gate(x, a, b, dt):
    """Exponential-Euler gate update: exact for frozen V, always in [0, 1],
    unconditionally stable and clamp-free (keeps gradients alive)."""
    s = a + b
    x_inf = a / s
    return x_inf + (x - x_inf) * torch.exp(-dt * s)


def hh_substep(v, m, h, n, current, dt, g_na, g_k, g_l, e_na, e_k, e_l, inv_c):
    """One integration substep as a pure function (easy to ``torch.compile``)."""
    am, bm, ah, bh, an, bn = hh_rates(v)
    m = _gate(m, am, bm, dt)
    h = _gate(h, ah, bh, dt)
    n = _gate(n, an, bn, dt)
    n2 = n * n
    i_ion = (g_na * m * m * m * h) * (v - e_na) + (g_k * n2 * n2) * (v - e_k) + g_l * (v - e_l)
    v = v + (dt * inv_c) * (current - i_ion)
    return v, m, h, n


class HodgkinHuxleySoma(BaseSoma):
    """Vectorized Hodgkin-Huxley neuron.

    Units: mV, ms, uA/cm^2, mS/cm^2 (classic shifted convention, rest at 0).
    One ``step`` advances ``dt`` ms using ``substeps`` Euler substeps for V and
    exponential-Euler for the gates (rates recomputed every substep).
    A spike is emitted once per upward crossing of ``threshold`` (the peak
    within the step is tracked, so a fast excursion inside one step is not
    missed). HH has no artificial refractory period.

    ``step`` calls ``self._substep``; assign ``torch.compile(hh_substep)`` to it
    to fuse the substep on GPU.
    """

    _state_names = BaseSoma._state_names + ("m", "h", "n", "was_above")

    def __init__(
        self,
        E_K=-12.0,
        E_Na=115.0,
        E_L=10.613,
        g_K=36.0,
        g_Na=120.0,
        g_L=0.3,
        C_m=1.0,
        dt=0.1,
        substeps=10,
        threshold=80.0,
        surrogate_slope=10.0,
        surrogate=True,
        device=None,
        dtype=torch.float32,
        v_init=0.0,
    ):
        super().__init__(
            refractory_period=0, device=device, dtype=dtype,
            surrogate_slope=surrogate_slope, surrogate=surrogate, v_init=v_init,
        )
        if C_m <= 0 or dt <= 0 or substeps < 1:
            raise ValueError("C_m, dt must be > 0 and substeps >= 1")
        self.E_K, self.E_Na, self.E_L = float(E_K), float(E_Na), float(E_L)
        self.g_K, self.g_Na, self.g_L = float(g_K), float(g_Na), float(g_L)
        self.C_m, self.dt, self.substeps = float(C_m), float(dt), int(substeps)
        self.threshold = float(threshold)
        self._substep = hh_substep
        self.initialize_state()

    def _default_states(self, like, voltage=None):
        states = super()._default_states(like, voltage)
        am, bm, ah, bh, an, bn = hh_rates(states["V"])
        states["m"] = am / (am + bm)  # steady state at the initial voltage
        states["h"] = ah / (ah + bh)
        states["n"] = an / (an + bn)
        states["was_above"] = states["V"] >= self.threshold
        return states

    def step(self, current):
        current = self._prepare_input(current)
        self._ensure_state(current)

        dt = self.dt / self.substeps
        consts = (self.g_Na, self.g_K, self.g_L, self.E_Na, self.E_K, self.E_L,
                  1.0 / self.C_m)
        v, m, h, n = self.V, self.m, self.h, self.n
        peak = v
        for _ in range(self.substeps):
            v, m, h, n = self._substep(v, m, h, n, current, dt, *consts)
            peak = torch.maximum(peak, v)

        crossing = (peak >= self.threshold) & ~self.was_above
        spike = spike_fn(peak - self.threshold, self.surrogate_slope, self.use_surrogate)
        spike = spike.masked_fill(~crossing, 0.0)

        self.V, self.m, self.h, self.n = v, m, h, n
        self.was_above = (v >= self.threshold).detach()
        return spike