from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from core.synapse import FastSynapseGroup


@dataclass(frozen=True)
class Connection:
    """Thông tin của một kết nối giữa hai neuron."""

    pre_neuron: Any
    post_neuron: Any
    synapse_idx: int
    axon_idx: int
    weight: float
    delay: int


class Connector:
    """
    Quản lý kết nối giữa các neuron thông qua một FastSynapseGroup.

    synapse_idx:
        Chỉ số trong FastSynapseGroup, dùng cho Dendrite.

    axon_idx:
        Chỉ số cục bộ trong Axon của neuron trước.
    """

    def __init__(self, synapse_group: FastSynapseGroup):
        self.synapse_group = synapse_group
        self.connections: list[Connection] = []
        self._next_index = 0
        self._built = False

    def connect(
        self,
        pre_neuron: Any,
        post_neuron: Any,
        *,
        weight: float = 0.5,
        delay: int = 1,
    ) -> Connection:
        """Tạo một kết nối từ pre_neuron đến post_neuron."""

        if self._built:
            raise RuntimeError(
                "Cannot connect after Connector.build()."
            )

        if self._next_index >= self.synapse_group.num_synapses:
            raise RuntimeError(
                "FastSynapseGroup has no remaining synapse capacity."
            )

        if not isfinite(weight):
            raise ValueError("weight must be finite.")

        if not 1 <= delay <= self.synapse_group.max_delay:
            raise ValueError(
                "delay must be between 1 and max_delay."
            )

        if not hasattr(pre_neuron, "axon"):
            raise TypeError("pre_neuron must have an axon.")

        if not hasattr(post_neuron, "dendrite"):
            raise TypeError("post_neuron must have a dendrite.")

        synapse_idx = self._next_index

        # Đăng ký synapse vào Axon của neuron trước.
        axon_idx = pre_neuron.axon.branch(
            weight=weight,
        )

        # Đăng ký chỉ số toàn cục vào Dendrite neuron sau.
        post_neuron.dendrite.attach(synapse_idx)

        # Cập nhật thông số của FastSynapseGroup.
        self.synapse_group.weights[synapse_idx] = weight
        self.synapse_group.delays[synapse_idx] = delay

        connection = Connection(
            pre_neuron=pre_neuron,
            post_neuron=post_neuron,
            synapse_idx=synapse_idx,
            axon_idx=axon_idx,
            weight=weight,
            delay=delay,
        )

        self.connections.append(connection)
        self._next_index += 1

        return connection

    def build(self) -> None:
        """
        Hoàn tất cấu trúc kết nối.

        Phải gọi sau khi đã tạo tất cả các kết nối.
        """
        if self._built:
            return

        # Mỗi Dendrite chỉ build một lần.
        dendrites = {
            id(connection.post_neuron.dendrite):
            connection.post_neuron.dendrite
            for connection in self.connections
        }

        for dendrite in dendrites.values():
            dendrite.build()

        self._built = True

    @property
    def num_connections(self) -> int:
        return self._next_index

    def reset(self) -> None:
        """Reset trạng thái động của nhóm synapse."""
        self.synapse_group.reset_state()