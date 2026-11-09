from __future__ import annotations

import torch
import torch.nn as nn


class Axon(nn.Module):
    """
    Vectorized Axon managing multiple synapses.

    - Preallocated weight storage.
    - Geometric capacity growth.
    - No Python loop during transmission.
    - Supports CPU and CUDA.
    - Preserves autograd path through synaptic states.
    """

    def __init__(
        self,
        *,
        capacity: int = 16,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if capacity < 1:
            raise ValueError("capacity must be >= 1.")

        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point.")

        self.capacity = capacity
        self.count = 0

        self.register_buffer(
            "_weights",
            torch.empty(capacity, device=device, dtype=dtype),
            persistent=True,
        )

        self.register_buffer(
            "_states",
            torch.empty(0, device=device, dtype=dtype),
            persistent=False,
        )

    @property
    def weights(self) -> torch.Tensor:
        return self._weights[:self.count]

    @property
    def states(self) -> torch.Tensor:
        return self._states

    def branch(
        self,
        weight: float,
        initial_state: float = 0.0,
    ) -> int:
        """Add one synapse."""

        if self.count == self.capacity:
            new_capacity = self.capacity * 2

            new_weights = torch.empty(
                (int(new_capacity),),
                device=self._weights.device,
                dtype=self._weights.dtype,
            )


            with torch.no_grad():
                new_weights[:self.count].copy_(self.weights)

            self._weights = new_weights
            self.capacity = new_capacity

        index = self.count

        with torch.no_grad():
            self._weights[index] = weight

        new_state = torch.tensor(
            [initial_state],
            device=self._states.device,
            dtype=self._states.dtype,
        )

        self._states = torch.cat(
            (self._states, new_state)
        )

        self.count += 1

        return index

    def transmit(
        self,
        spike: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update all synaptic states simultaneously.

        spike must be scalar or broadcastable to [count].
        """

        if self.count == 0:
            return self._states

        spike = torch.as_tensor(
            spike,
            device=self._states.device,
            dtype=self._states.dtype,
        )

        try:
            contribution = spike * self.weights
        except RuntimeError as exc:
            raise ValueError(
                "spike must be broadcastable to synapse shape."
            ) from exc

        self._states = self._states + contribution

        return self._states

    def reset_state(self, value: float = 0.0) -> None:
        self._states = torch.full_like(
            self._states,
            value,
        )