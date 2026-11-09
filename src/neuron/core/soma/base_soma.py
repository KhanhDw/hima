
# base_soma.py

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from .surrogate import spike_fn


TensorLike = Union[torch.Tensor, float, int]
ShapeLike = Union[int, Sequence[int], torch.Size]
DeviceLike = Optional[Union[str, torch.device]]


class BaseSoma(nn.Module):
    """
    Base class for a spiking-neuron soma.

    Runtime states:
        V:
            Membrane potential.

        refractory_left:
            Number of remaining refractory steps.

    Subclasses should:
        1. Override _state_names if they introduce additional states.
        2. Override _default_states if they introduce additional states.
        3. Implement step().

    Notes:
        - Runtime states are registered as non-persistent buffers.
        - Runtime states are therefore excluded from state_dict().
        - V.dtype and V.device are the source of truth for computation.
    """

    V: torch.Tensor
    refractory_left: torch.Tensor

    _state_names = (
        "V",
        "refractory_left",
    )

    def __init__(
        self,
        *,
        refractory_period: int = 2,
        surrogate_slope: float = 10.0,
        surrogate: bool = True,
        v_init: float = 0.0,
        device: DeviceLike = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # --------------------------------------------------
        # Validate configuration
        # --------------------------------------------------

        if isinstance(refractory_period, bool) or not isinstance(
            refractory_period, int
        ):
            raise TypeError("refractory_period must be an integer.")

        if refractory_period < 0:
            raise ValueError("refractory_period must be >= 0.")

        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype.")

        if not dtype.is_floating_point:
            raise TypeError("Soma computation dtype must be floating point.")

        if surrogate_slope <= 0:
            raise ValueError("surrogate_slope must be > 0.")

        # --------------------------------------------------
        # Configuration
        # --------------------------------------------------

        self.refractory_period = refractory_period
        self.surrogate_slope = float(surrogate_slope)
        self.use_surrogate = bool(surrogate)
        self.v_init = float(v_init)

        # --------------------------------------------------
        # Initial device
        # --------------------------------------------------

        initial_device: torch.device

        if device is None:
            initial_device = torch.device("cpu")
        else:
            initial_device = torch.device(device)

        # --------------------------------------------------
        # Runtime buffers
        # --------------------------------------------------

        self.register_buffer(
            "V",
            torch.empty(
                (),
                device=initial_device,
                dtype=dtype,
            ).fill_(self.v_init),
            persistent=False,
        )

        self.register_buffer(
            "refractory_left",
            torch.zeros(
                (),
                device=initial_device,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        self._state_initialized = True

    # ======================================================
    # Shape utilities
    # ======================================================

    @staticmethod
    def _normalize_shape(shape: ShapeLike) -> tuple[int, ...]:
        """
        Convert shape to a normalized tuple.

        Examples:
            10       -> (10,)
            (2, 10)  -> (2, 10)
            torch.Size([2, 10]) -> (2, 10)
            ()       -> ()
        """

        if isinstance(shape, int):
            normalized = (shape,)
        else:
            normalized = tuple(shape)

        for dimension in normalized:
            if not isinstance(dimension, int):
                raise TypeError("Shape dimensions must be integers.")

            if dimension < 0:
                raise ValueError("Shape dimensions must be >= 0.")

        return normalized

    # ======================================================
    # State creation
    # ======================================================

    def _default_states(
        self,
        like: torch.Tensor,
        voltage: Optional[TensorLike] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Create default runtime states.

        Subclasses can override this method to add more states.
        """

        if voltage is None:
            membrane = torch.full_like(
                like,
                fill_value=self.v_init,
            )
        else:
            membrane = torch.as_tensor(
                voltage,
                device=like.device,
                dtype=like.dtype,
            )

            membrane = torch.broadcast_to(
                membrane,
                like.shape,
            ).clone()

        refractory = torch.zeros(
            like.shape,
            device=like.device,
            dtype=torch.int32,
        )

        return {
            "V": membrane,
            "refractory_left": refractory,
        }

    def _set_states(
        self,
        states: Dict[str, torch.Tensor],
    ) -> None:
        """
        Update registered runtime buffers.

        Buffer names must match _state_names exactly.
        """

        expected = set(self._state_names)
        received = set(states.keys())

        if expected != received:
            raise ValueError(
                f"Invalid state keys. "
                f"Expected {expected}, received {received}."
            )

        for name in self._state_names:
            value = states[name]

            if not isinstance(value, torch.Tensor):
                raise TypeError(f"State '{name}' must be a Tensor.")

            setattr(self, name, value)

    # ======================================================
    # Initialization
    # ======================================================

    def initialize_state(
        self,
        shape: ShapeLike = (),
        *,
        device: DeviceLike = None,
        dtype: Optional[torch.dtype] = None,
        voltage: Optional[TensorLike] = None,
    ) -> None:
        """
        Initialize neuron states.

        Args:
            shape:
                Shape of the neuron population.

            device:
                Target device. Defaults to current V.device.

            dtype:
                Target floating-point dtype. Defaults to V.dtype.

            voltage:
                Optional initial membrane potential.
        """

        normalized_shape = self._normalize_shape(shape)

        target_device: torch.device

        if device is None:
            target_device = self.V.device
        else:
            target_device = torch.device(device)

        target_dtype = self.V.dtype if dtype is None else dtype

        if not target_dtype.is_floating_point:
            raise TypeError("State dtype must be floating point.")

        reference = torch.empty(
            size=normalized_shape,
            device=target_device,
            dtype=target_dtype,
        )

        states = self._default_states(
            reference,
            voltage=voltage,
        )

        self._set_states(states)
        self._state_initialized = True

    # ======================================================
    # Input preparation
    # ======================================================

    def _prepare_input(
        self,
        input_current: TensorLike,
    ) -> torch.Tensor:
        """
        Convert input to a tensor.

        Tensor input:
            Preserves its device and casts its dtype.

        Scalar input:
            Uses the current state device.
        """

        if isinstance(input_current, torch.Tensor):
            return input_current.to(
                dtype=self.V.dtype,
            )

        return torch.as_tensor(
            input_current,
            device=self.V.device,
            dtype=self.V.dtype,
        )

    # ======================================================
    # State synchronization
    # ======================================================

    def _ensure_state(
        self,
        reference: torch.Tensor,
    ) -> None:
        """
        Ensure runtime state has the same shape and device
        as the input tensor.

        Existing state values are preserved when possible.
        """

        if not isinstance(reference, torch.Tensor):
            raise TypeError("reference must be a Tensor.")

        target_shape = reference.shape
        target_device = reference.device
        target_dtype = reference.dtype

        state_shape_changed = self.V.shape != target_shape

        if state_shape_changed:
            self.initialize_state(
                shape=target_shape,
                device=target_device,
                dtype=target_dtype,
            )
            return

        device_changed = self.V.device != target_device
        dtype_changed = self.V.dtype != target_dtype

        if device_changed or dtype_changed:
            self.V = self.V.to(
                device=target_device,
                dtype=target_dtype,
            )

            self.refractory_left = self.refractory_left.to(
                device=target_device,
                dtype=torch.int32,
            )

    # ======================================================
    # Reset
    # ======================================================

    def reset_state(
        self,
        mask: Optional[torch.Tensor] = None,
        *,
        voltage: Optional[TensorLike] = None,
    ) -> None:
        """
        Reset all neurons or selected neurons.

        Args:
            mask:
                Boolean tensor. True indicates neurons to reset.

            voltage:
                Optional reset membrane potential.
        """

        fresh_states = self._default_states(
            self.V,
            voltage=voltage,
        )

        if mask is None:
            self._set_states(fresh_states)
            return

        reset_mask = torch.as_tensor(
            mask,
            device=self.V.device,
            dtype=torch.bool,
        )

        try:
            reset_mask = torch.broadcast_to(
                reset_mask,
                self.V.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "mask cannot be broadcast to the state shape."
            ) from exc

        self.V = torch.where(
            reset_mask,
            fresh_states["V"],
            self.V,
        )

        self.refractory_left = torch.where(
            reset_mask,
            fresh_states["refractory_left"],
            self.refractory_left,
        )

    # ======================================================
    # Runtime state management
    # ======================================================

    def detach_state(self) -> None:
        """
        Detach runtime states from the previous computation graph.

        Useful for truncated backpropagation through time.
        """

        self.V = self.V.detach()
        self.refractory_left = self.refractory_left.detach()

    def get_runtime_state(self) -> Dict[str, torch.Tensor]:
        """
        Return a detached copy of runtime states.
        """

        return {
            name: getattr(self, name).detach().clone()
            for name in self._state_names
        }

    def load_runtime_state(
        self,
        states: Dict[str, torch.Tensor],
    ) -> None:
        """
        Restore runtime states.

        State tensors are moved to the current V device.
        """

        expected = set(self._state_names)
        received = set(states.keys())

        if expected != received:
            raise ValueError(
                f"Invalid state keys. "
                f"Expected {expected}, received {received}."
            )

        restored: Dict[str, torch.Tensor] = {}

        for name in self._state_names:
            value = states[name]

            if not isinstance(value, torch.Tensor):
                raise TypeError(f"State '{name}' must be a Tensor.")

            if name == "V":
                restored[name] = value.to(
                    device=self.V.device,
                    dtype=self.V.dtype,
                )
            else:
                restored[name] = value.to(
                    device=self.V.device,
                    dtype=torch.int32,
                )

        self._set_states(restored)

    # ======================================================
    # Refractory state
    # ======================================================

    @property
    def is_refractory(self) -> torch.Tensor:
        """
        Boolean mask indicating refractory neurons.
        """

        return self.refractory_left > 0

    @property
    def active_mask(self) -> torch.Tensor:
        """
        Boolean mask indicating active neurons.
        """

        return self.refractory_left <= 0

    def update_refractory(
        self,
        spike: torch.Tensor,
    ) -> None:
        """
        Update refractory counters.

        Call this once per simulation step.

        Existing refractory counters are decremented first.
        Newly spiking neurons receive the full refractory period.
        """

        spike_mask = spike.detach().to(
            device=self.V.device,
            dtype=torch.bool,
        )

        self.refractory_left = torch.clamp(
            self.refractory_left - 1,
            min=0,
        )

        if self.refractory_period > 0:
            self.refractory_left = torch.where(
                spike_mask,
                torch.full_like(
                    self.refractory_left,
                    self.refractory_period,
                ),
                self.refractory_left,
            )

    # ======================================================
    # Spike generation
    # ======================================================

    def _fire(
        self,
        voltage: torch.Tensor,
        threshold: TensorLike,
    ) -> torch.Tensor:
        """
        Generate spikes using surrogate gradients.

        Refractory neurons cannot spike.
        """

        threshold_tensor = torch.as_tensor(
            threshold,
            device=voltage.device,
            dtype=voltage.dtype,
        )

        spike = spike_fn(
            voltage - threshold_tensor,
            self.surrogate_slope,
            self.use_surrogate,
        )

        return spike.masked_fill(
            self.is_refractory,
            0.0,
        )

    # ======================================================
    # Simulation interface
    # ======================================================

    def step(
        self,
        input_current: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance the neuron by one simulation step.

        Must be implemented by subclasses.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement step()."
        )

    def forward(
        self,
        input_current: TensorLike,
    ) -> torch.Tensor:
        """
        One simulation time step.
        """

        current = self._prepare_input(input_current)

        self._ensure_state(current)

        return self.step(current)
