from __future__ import annotations

import torch


class SimulationClock:

    def __init__(
        self,
        *,
        device: str = "cuda",
    ):

        self.device = torch.device(
            device if (
                device != "cuda" or torch.cuda.is_available()
            ) else "cpu"
        )

        self.time = torch.zeros(
            (),
            dtype=torch.int64,
            device=self.device,
        )

    @torch.no_grad()
    def tick(self) -> torch.Tensor:
        """Advance simulation time by one step."""

        self.time.add_(1)

        return self.time

    @torch.no_grad()
    def reset(self) -> None:
        """Reset simulation time."""

        self.time.zero_()

    def get_time(self) -> torch.Tensor:
        """Return current simulation time."""

        return self.time