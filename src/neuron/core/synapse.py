from __future__ import annotations

import torch


class FastSynapseGroup:

    def __init__(
        self,
        num_synapses: int,
        *,
        max_delay: int = 32,
        learning_rate: float = 0.01,
        trace_decay: float = 0.95,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):

        if num_synapses < 1:
            raise ValueError("num_synapses must be >= 1.")

        if max_delay < 1:
            raise ValueError("max_delay must be >= 1.")

        if not 0.0 <= trace_decay <= 1.0:
            raise ValueError("trace_decay must be in [0, 1].")

        if learning_rate < 0:
            raise ValueError("learning_rate must be >= 0.")

        self.device = torch.device(
            device if (
                device != "cuda" or torch.cuda.is_available()
            ) else "cpu"
        )

        self.dtype = dtype
        self.num_synapses = num_synapses
        self.max_delay = max_delay

        self.learning_rate = learning_rate
        self.trace_decay = trace_decay

        self.weights = torch.full(
            (num_synapses,),
            0.5,
            device=self.device,
            dtype=dtype,
        )

        self.delays = torch.randint(
            1,
            max_delay + 1,
            (num_synapses,),
            device=self.device,
            dtype=torch.int64,
        )

        self.delay_buffer = torch.zeros(
            max_delay,
            num_synapses,
            device=self.device,
            dtype=dtype,
        )

        self.last_output = torch.zeros(
            num_synapses,
            device=self.device,
            dtype=dtype,
        )

        self.eligibility_traces = torch.zeros_like(
            self.weights
        )

        self.buffer_idx = 0

    @torch.no_grad()
    def receive(self, spikes: torch.Tensor):

        spikes = torch.as_tensor(
            spikes,
            device=self.device,
            dtype=self.dtype,
        )

        if spikes.numel() == 1:
            spikes = spikes.expand(self.num_synapses)

        if spikes.shape != (self.num_synapses,):
            raise ValueError(
                "spikes must have shape [num_synapses]."
            )

        target_indices = (
            self.buffer_idx + self.delays
        ) % self.max_delay

        self.delay_buffer.scatter_add_(
            0,
            target_indices.unsqueeze(0),
            spikes.unsqueeze(0),
        )

    @torch.no_grad()
    def transmit(self):

        current_spikes = self.delay_buffer[
            self.buffer_idx
        ]

        self.last_output.copy_(
            current_spikes * self.weights
        )

        self.delay_buffer[
            self.buffer_idx
        ].zero_()

        self.buffer_idx = (
            self.buffer_idx + 1
        ) % self.max_delay

        return self.last_output

    @torch.no_grad()
    def update_trace(
        self,
        pre_spiked: torch.Tensor,
        post_spiked: torch.Tensor,
    ):

        pre = torch.as_tensor(
            pre_spiked,
            device=self.device,
            dtype=torch.bool,
        )

        post = torch.as_tensor(
            post_spiked,
            device=self.device,
            dtype=torch.bool,
        )

        active_mask = pre | post

        self.eligibility_traces.mul_(
            self.trace_decay
        )

        self.eligibility_traces.add_(
            active_mask.to(self.dtype)
        )

    @torch.no_grad()
    def apply_reward_modulation(
        self,
        reward: float,
    ):

        self.weights.add_(
            self.eligibility_traces,
            alpha=self.learning_rate * reward,
        )

        self.weights.clamp_(0.0, 5.0)

    @torch.no_grad()
    def reset_state(self):

        self.delay_buffer.zero_()
        self.last_output.zero_()
        self.eligibility_traces.zero_()

        self.buffer_idx = 0

    @torch.no_grad()
    def clamp(self):

        self.weights.clamp_(0.0, 5.0)