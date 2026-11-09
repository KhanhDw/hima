
from typing import Union

import torch


TimeType = Union[int, torch.Tensor]


class InputNeuronGroup:
    def __init__(
        self,
        num_neurons: int,
        rate_alpha: float = 0.1,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        if num_neurons <= 0:
            raise ValueError("num_neurons must be greater than 0")

        if not 0.0 <= rate_alpha <= 1.0:
            raise ValueError("rate_alpha must be between 0 and 1")

        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"

        self.device = torch.device(device)
        self.dtype = dtype

        self.num_neurons = num_neurons
        self.rate_alpha = rate_alpha

        # Runtime statistics
        self.steps = torch.zeros(
            num_neurons,
            dtype=torch.int64,
            device=self.device,
        )

        self.fire_count = torch.zeros(
            num_neurons,
            dtype=torch.int64,
            device=self.device,
        )

        self.last_spike_time = torch.full(
            (num_neurons,),
            -1,
            dtype=torch.int64,
            device=self.device,
        )

        self.firing_rate = torch.zeros(
            num_neurons,
            dtype=self.dtype,
            device=self.device,
        )

    @torch.no_grad()
    def stimulate(
        self,
        spikes: torch.Tensor,
        time: TimeType,
    ) -> torch.Tensor:
        """
        spikes:
            Tensor shape (num_neurons,).
            Giá trị spike thường là 0 hoặc 1.

        time:
            Python int hoặc scalar Tensor.
        """

        # 1. Kiểm tra đầu vào
        if spikes.ndim != 1:
            raise ValueError("spikes must be a 1D Tensor")

        if spikes.numel() != self.num_neurons:
            raise ValueError(
                f"Expected {self.num_neurons} spikes, "
                f"got {spikes.numel()}"
            )

        if spikes.device != self.device:
            raise ValueError(
                "spikes and InputNeuronGroup must use the same device"
            )

        if spikes.dtype != self.dtype:
            raise ValueError(
                "spikes and InputNeuronGroup must use the same dtype"
            )

        # 2. Chuẩn hóa timestep
        time_tensor = torch.as_tensor(
            time,
            dtype=torch.int64,
            device=self.device,
        )

        if time_tensor.numel() != 1:
            raise ValueError("time must be a scalar")

        time_tensor = time_tensor.reshape(())

        # 3. Xác định neuron phát spike
        active_mask = spikes > 0

        # 4. Cập nhật số bước
        self.steps.add_(1)

        # 5. Cập nhật số lần phát spike
        self.fire_count.add_(active_mask)

        # 6. Cập nhật thời điểm spike gần nhất
        self.last_spike_time.copy_(
            torch.where(
                active_mask,
                time_tensor,
                self.last_spike_time,
            )
        )

        # 7. Cập nhật firing rate
        self.firing_rate.mul_(1.0 - self.rate_alpha)
        self.firing_rate.add_(
            spikes,
            alpha=self.rate_alpha,
        )

        return spikes

    @torch.no_grad()
    def reset(self):
        """Reset toàn bộ trạng thái nhóm neuron."""

        self.steps.zero_()
        self.fire_count.zero_()
        self.last_spike_time.fill_(-1)
        self.firing_rate.zero_()

        return self
