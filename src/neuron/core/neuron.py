
from typing import Union

import torch

from .axon import Axon
from .soma import LIFSoma
from .dendrite import Dendrite


TimeType = Union[int, torch.Tensor]


class Neuron:
    def __init__(
        self,
        theta: float = 1.0,
        leak: float = 0.9,
        refractory_period: int = 2,
        rate_alpha: float = 0.01,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        # Device và dtype thống nhất
        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"

        self.device = torch.device(device)
        self.dtype = dtype

        if not 0.0 <= rate_alpha <= 1.0:
            raise ValueError("rate_alpha must be between 0 and 1")

        self.rate_alpha = rate_alpha

        # Các thành phần
        self.axon = Axon(
            device=self.device,
            dtype=self.dtype,
        )

        self.dendrite = Dendrite(device=str(self.device))

        self.soma = LIFSoma(
            theta=theta,
            leak=leak,
            refractory_period=refractory_period,
        ).to(
            device=self.device,
            dtype=self.dtype,
        )

        # Thống kê
        self.steps = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )

        self.fire_count = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )

        self.last_spike_time = torch.full(
            (), -1, dtype=torch.int64, device=self.device
        )

        self.firing_rate = torch.zeros(
            (), dtype=self.dtype, device=self.device
        )

        self._built = False

    def build(self):
        if self._built:
            return self

        self.dendrite.build()
        self._built = True

        return self

    def step(
        self,
        synapse_outputs: torch.Tensor,
        time: TimeType,
    ) -> torch.Tensor:

        if not self._built:
            raise RuntimeError("Call build() before simulation")

        if synapse_outputs.device != self.device:
            raise ValueError(
                "synapse_outputs and Neuron must use the same device"
            )

        if synapse_outputs.dtype != self.dtype:
            raise ValueError(
                "synapse_outputs and Neuron must use the same dtype"
            )

        # 1. Dendrite tổng hợp dòng điện
        current = self.dendrite.current(synapse_outputs)

        # 2. Soma xử lý điện thế và tạo spike
        spike = self.soma.step(current)

        if spike.numel() != 1:
            raise ValueError(
                "Neuron expects exactly one scalar spike"
            )

        spike = spike.reshape(())

        # 3. Axon truyền tín hiệu
        # Không dùng if spike: tránh đồng bộ GPU-CPU.
        self.axon.transmit(spike)

        # 4. Cập nhật firing rate
        # Đây là thống kê không tham gia gradient.
        with torch.no_grad():
            spike_value = spike.detach()

            self.firing_rate.mul_(1.0 - self.rate_alpha)
            self.firing_rate.add_(
                spike_value * self.rate_alpha
            )

            # 5. Cập nhật thống kê
            if isinstance(time, torch.Tensor):
                time_tensor = time.to(
                    device=self.device,
                    dtype=torch.int64,
                )
            else:
                time_tensor = torch.tensor(
                    time,
                    dtype=torch.int64,
                    device=self.device,
                )

            if time_tensor.numel() != 1:
                raise ValueError("time must be a scalar")

            self.steps.add_(1)

            self.fire_count.add_(
                spike_value.to(torch.int64)
            )

            self.last_spike_time.copy_(
                torch.where(
                    spike_value > 0,
                    time_tensor.reshape(()),
                    self.last_spike_time,
                )
            )

        return spike

    def reset(self):
        self.soma.reset_state()
        self.axon.reset_state()

        with torch.no_grad():
            self.steps.zero_()
            self.fire_count.zero_()
            self.last_spike_time.fill_(-1)
            self.firing_rate.zero_()

        return self
