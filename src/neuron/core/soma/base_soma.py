from __future__ import annotations

from typing import Dict, Optional

import torch

from .surrogate import spike_fn


class BaseSoma:
    """Shared state handling for SNN soma models.

    Conventions
    -----------
    * Every state tensor has the shape of the input (one value per neuron).
      A *shape change* starts a fresh state; a *device change* migrates the
      state (values preserved). The state dtype is ``self.dtype`` (configured
      once, e.g. float32 even under autocast); inputs are cast to it.
    * Subclasses list their tensors in ``_state_names`` and describe their
      initial values in ``_default_states``. ``reset``, ``detach_state``,
      ``to``, ``state_dict`` and shape/device handling are then generic, so a
      new state variable can never be "forgotten" by one of them.
    * Refractory: after a spike the neuron is silent for exactly
      ``refractory_period`` steps (membrane held, spikes masked).
    * Leaf subclasses call ``self.initialize_state()`` at the end of
      ``__init__`` (after their own hyper-parameters exist).
    """

    _state_names = ("V", "refractory_left")

    def __init__(
        self,
        refractory_period: int = 2,
        device=None,
        dtype: torch.dtype = torch.float32,
        surrogate_slope: float = 10.0,
        surrogate: bool = True,
        v_init: float = 0.0,
    ):
        if int(refractory_period) < 0:
            raise ValueError("refractory_period must be >= 0")
        if not dtype.is_floating_point:
            raise TypeError("Soma state dtype must be floating point")
        if surrogate_slope <= 0:
            raise ValueError("surrogate_slope must be > 0")

        self.refractory_period = int(refractory_period)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.surrogate_slope = float(surrogate_slope)
        self.use_surrogate = bool(surrogate)
        self.v_init = float(v_init)

        # Scalar placeholders until the leaf class calls initialize_state().
        self.V = torch.zeros((), device=self.device, dtype=self.dtype)
        self.refractory_left = torch.zeros((), device=self.device, dtype=torch.int32)

    # ------------------------------------------------------------- state ---
    def _default_states(self, like: torch.Tensor, voltage: Optional[float] = None
                        ) -> Dict[str, torch.Tensor]:
        """Fresh state with the shape/device/dtype of ``like``."""
        v0 = self.v_init if voltage is None else float(voltage)
        return {
            "V": torch.full_like(like, v0),
            "refractory_left": torch.zeros(like.shape, device=like.device, dtype=torch.int32),
        }

    def _set_states(self, states: Dict[str, torch.Tensor]) -> None:
        for name, value in states.items():
            setattr(self, name, value)

    def _cast_states(self, device, dtype) -> None:
        for name in self._state_names:
            t = getattr(self, name)
            target = dtype if t.is_floating_point() else t.dtype
            setattr(self, name, t.to(device=device, dtype=target))

    def initialize_state(self, shape=(), device=None, dtype=None, voltage=None):
        device = torch.device(device) if device is not None else self.device
        dtype = dtype or self.dtype
        if not dtype.is_floating_point:
            raise TypeError("Soma state dtype must be floating point")
        self.device, self.dtype = device, dtype
        like = torch.empty(shape, device=device, dtype=dtype)
        self._set_states(self._default_states(like, voltage))
        return self

    def _prepare_input(self, current) -> torch.Tensor:
        if isinstance(current, torch.Tensor):
            return current if current.dtype == self.dtype else current.to(self.dtype)
        return torch.as_tensor(current, device=self.device, dtype=self.dtype)

    def _ensure_state(self, current: torch.Tensor) -> None:
        V = self.V
        if V.shape != current.shape:
            self._set_states(self._default_states(current))
            self.device, self.dtype = current.device, current.dtype
        elif V.device != current.device or V.dtype != current.dtype:
            self._cast_states(current.device, current.dtype)
            self.device, self.dtype = current.device, current.dtype

    def reset(self, mask=None):
        """Reset all neurons, or only entries where ``mask`` is True."""
        fresh = self._default_states(self.V)
        if mask is None:
            self._set_states(fresh)
            return self
        mask = torch.as_tensor(mask, device=self.V.device, dtype=torch.bool)
        if torch.broadcast_shapes(mask.shape, self.V.shape) != self.V.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} is not broadcastable to state "
                f"shape {tuple(self.V.shape)}"
            )
        for name, new in fresh.items():
            setattr(self, name, torch.where(mask, new, getattr(self, name)))
        return self

    def detach_state(self):
        """Cut the autograd graph at a truncated-BPTT boundary."""
        for name in self._state_names:
            setattr(self, name, getattr(self, name).detach())
        return self

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError("Soma state dtype must be floating point")
            self.dtype = dtype
        self._cast_states(self.device, self.dtype)
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {n: getattr(self, n).detach().clone() for n in self._state_names}

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        missing = set(self._state_names) - set(state)
        if missing:
            raise KeyError(f"missing state entries: {sorted(missing)}")
        for n in self._state_names:
            setattr(self, n, state[n].detach().clone())
        self.device, self.dtype = self.V.device, self.V.dtype
        return self

    # -------------------------------------------------------- refractory ---
    def active_mask(self) -> torch.Tensor:
        return self.refractory_left == 0

    def is_refractory(self) -> torch.Tensor:
        return self.refractory_left > 0

    def _refractory_mask(self) -> Optional[torch.Tensor]:
        """None when refractory is disabled -> callers skip all masking."""
        return (self.refractory_left > 0) if self.refractory_period else None

    def update_refractory(self, fired: torch.Tensor) -> None:
        """Advance one step; a new spike starts a fresh refractory interval."""
        if not self.refractory_period:
            return
        if fired.dtype != torch.bool:
            fired = fired != 0
        countdown = (self.refractory_left - 1).clamp_(min=0)
        self.refractory_left = countdown.masked_fill_(fired, self.refractory_period)

    # ------------------------------------------------------------- spike ---
    def _fire(self, voltage, threshold, refractory):
        spike = spike_fn(voltage - threshold, self.surrogate_slope, self.use_surrogate)
        if refractory is not None:
            spike = spike.masked_fill(refractory, 0.0)
        return spike

    def step(self, current):  # pragma: no cover
        raise NotImplementedError

    def forward(self, current):
        return self.step(current)

    __call__ = forward