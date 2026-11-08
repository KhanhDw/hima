import math


class RewardModulatedSTDP:

    def __init__(
        self,
        A_plus=0.005,
        A_minus=0.005,
        tau=20.0,
        reward_scale=1.0,
        weight_decay=0.0005
    ):
        self.A_plus = A_plus
        self.A_minus = A_minus

        self.tau = tau

        self.reward_scale = (
            reward_scale
        )

        self.weight_decay = (
            weight_decay
        )

    def update(
        self,
        synapse,
        post_spike_time,
        reward
    ):

        if not synapse.pre_spikes:
            return

        pre_spike_time = (
            synapse.pre_spikes[-1]
        )

        dt = (
            post_spike_time
            - pre_spike_time
        )

        stdp_term = 0.0

        if dt > 0:

            stdp_term = (
                self.A_plus
                * math.exp(
                    -dt / self.tau
                )
            )

        elif dt < 0:

            stdp_term = -(
                self.A_minus
                * math.exp(
                    dt / self.tau
                )
            )

        dw = (
            reward
            * self.reward_scale
            * stdp_term
        )

        synapse.weight += dw

        synapse.weight *= (
            1.0
            -
            self.weight_decay
        )

        synapse.clamp()

        return dw