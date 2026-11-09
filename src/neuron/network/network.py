from typing import Optional

import torch

from core.input_neuron import InputNeuronGroup
from core.neuron import Neuron
from core.synapse import FastSynapseGroup


class Network:
    """
    SNN Network.

    Luồng xử lý:
        InputNeuronGroup
            -> FastSynapseGroup
            -> Dendrite
            -> LIFSoma
            -> Axon

    Quy ước:
        - Mỗi timestep nhận một vector input_spikes.
        - Các SynapseGroup nhận spike đầu vào.
        - transmit() trả về Tensor đầu ra của synapse.
        - Dendrite sử dụng synapse indices trên Tensor tổng hợp.
        - Mỗi Neuron phát ra một spike scalar.
    """

    def __init__(
        self,
        input_group: InputNeuronGroup,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"

        self.device = torch.device(device)
        self.dtype = dtype

        self.input_group = input_group

        if input_group.device != self.device:
            raise ValueError(
                "InputNeuronGroup and Network must use the same device"
            )

        self.synapse_groups: list[FastSynapseGroup] = []
        self.neurons: list[Neuron] = []

        self.time = 0
        self._built = False

        self.last_input_spikes: Optional[torch.Tensor] = None
        self.last_synapse_outputs: Optional[torch.Tensor] = None
        self.last_output_spikes: Optional[torch.Tensor] = None

    # --------------------------------------------------
    # BUILD NETWORK
    # --------------------------------------------------

    def add_synapse_group(self, group: FastSynapseGroup):
        if self._built:
            raise RuntimeError(
                "Cannot add synapse groups after build()"
            )

        self.synapse_groups.append(group)
        return group

    def add_neuron(self, neuron: Neuron):
        if self._built:
            raise RuntimeError(
                "Cannot add neurons after build()"
            )

        if neuron.device != self.device:
            raise ValueError(
                "Neuron and Network must use the same device"
            )

        if neuron.dtype != self.dtype:
            raise ValueError(
                "Neuron and Network must use the same dtype"
            )

        self.neurons.append(neuron)
        return neuron

    def build(self):
        """
        Hoàn tất cấu trúc mạng trước khi mô phỏng.
        """

        if self._built:
            return self

        if not self.neurons:
            raise RuntimeError(
                "Network must contain at least one neuron"
            )

        if not self.synapse_groups:
            raise RuntimeError(
                "Network must contain at least one synapse group"
            )

        for neuron in self.neurons:
            neuron.build()

        self._built = True
        return self

    # --------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------

    def _validate_input(
        self,
        input_spikes: torch.Tensor,
    ):
        if input_spikes.ndim != 1:
            raise ValueError(
                "input_spikes must be a 1D Tensor"
            )

        if input_spikes.numel() != self.input_group.num_neurons:
            raise ValueError(
                "Input spike count does not match InputNeuronGroup"
            )

        if input_spikes.device != self.device:
            raise ValueError(
                "Input spikes and Network must use the same device"
            )

        if input_spikes.dtype != self.dtype:
            raise ValueError(
                "Input spikes and Network must use the same dtype"
            )

    # --------------------------------------------------
    # SYNAPSE PROCESSING
    # --------------------------------------------------

    def _process_synapses(
        self,
        input_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Truyền spike qua tất cả SynapseGroup.

        Mỗi group phải có:
            receive(spikes)
            transmit() -> Tensor
        """

        outputs = []

        for group in self.synapse_groups:
            group.receive(input_spikes)

            synapse_output = group.transmit()

            if not isinstance(synapse_output, torch.Tensor):
                raise TypeError(
                    "FastSynapseGroup.transmit() must return a Tensor"
                )

            if synapse_output.ndim != 1:
                raise ValueError(
                    "Synapse output must be a 1D Tensor"
                )

            if synapse_output.device != self.device:
                raise ValueError(
                    "Synapse output is on the wrong device"
                )

            if synapse_output.dtype != self.dtype:
                raise ValueError(
                    "Synapse output has the wrong dtype"
                )

            outputs.append(synapse_output)

        return torch.cat(outputs, dim=0)

    # --------------------------------------------------
    # NETWORK STEP
    # --------------------------------------------------

    def step(
        self,
        input_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chạy một timestep của toàn mạng.

        Args:
            input_spikes:
                Tensor shape (num_input_neurons,).

        Returns:
            Tensor shape (num_neurons,).
        """

        if not self._built:
            raise RuntimeError(
                "Call build() before running the network"
            )

        self._validate_input(input_spikes)

        current_time = self.time

        # 1. Cập nhật trạng thái nhóm neuron đầu vào
        input_spikes = self.input_group.stimulate(
            input_spikes,
            current_time,
        )

        # 2. Truyền spike qua các synapse
        synapse_outputs = self._process_synapses(
            input_spikes
        )

        # 3. Xử lý toàn bộ neuron đích
        neuron_spikes = []

        for neuron in self.neurons:
            spike = neuron.step(
                synapse_outputs,
                current_time,
            )

            neuron_spikes.append(spike)

        # 4. Tổng hợp spike đầu ra
        output_spikes = torch.stack(
            neuron_spikes,
            dim=0,
        )

        # 5. Lưu kết quả timestep
        self.last_input_spikes = input_spikes
        self.last_synapse_outputs = synapse_outputs
        self.last_output_spikes = output_spikes

        self.time += 1

        return output_spikes

    # --------------------------------------------------
    # RUN SIMULATION
    # --------------------------------------------------

    def run(
        self,
        input_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chạy nhiều timestep.

        Args:
            input_sequence:
                Tensor shape (time_steps, num_input_neurons).

        Returns:
            Tensor shape (time_steps, num_output_neurons).
        """

        if not self._built:
            raise RuntimeError(
                "Call build() before running the network"
            )

        if input_sequence.ndim != 2:
            raise ValueError(
                "input_sequence must have shape "
                "(time_steps, num_input_neurons)"
            )

        if input_sequence.shape[1] != self.input_group.num_neurons:
            raise ValueError(
                "Input sequence has an incorrect neuron dimension"
            )

        if input_sequence.device != self.device:
            raise ValueError(
                "Input sequence and Network must use the same device"
            )

        if input_sequence.dtype != self.dtype:
            raise ValueError(
                "Input sequence and Network must use the same dtype"
            )

        outputs = []

        for t in range(input_sequence.shape[0]):
            output = self.step(input_sequence[t])
            outputs.append(output)

        return torch.stack(outputs, dim=0)

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self):
        """
        Reset trạng thái toàn mạng.
        """

        self.input_group.reset()

        for neuron in self.neurons:
            neuron.reset()

        for group in self.synapse_groups:
            reset_fn = getattr(group, "reset", None)

            if callable(reset_fn):
                reset_fn()

        self.time = 0

        self.last_input_spikes = None
        self.last_synapse_outputs = None
        self.last_output_spikes = None

        return self

    # --------------------------------------------------
    # INFORMATION
    # --------------------------------------------------

    @property
    def num_neurons(self) -> int:
        return len(self.neurons)

    @property
    def num_synapse_groups(self) -> int:
        return len(self.synapse_groups)

    def summary(self) -> dict[str, int]:
        return {
            "input_neurons": self.input_group.num_neurons,
            "neurons": self.num_neurons,
            "synapse_groups": self.num_synapse_groups,
            "time": self.time,
        }