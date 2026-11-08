import math


class STDP:

    def __init__(
        self,
        A_plus=0.005,
        A_minus=0.005,
        tau=20.0,
        weight_decay=0.0005
    ):
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau = tau

        self.weight_decay = weight_decay

    def update(
        self,
        synapse,
        post_spike_time
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

        if dt > 0:

            dw = (
                self.A_plus
                * math.exp(
                    -dt / self.tau
                )
            )

            synapse.weight += dw

        elif dt < 0:

            dw = (
                self.A_minus
                * math.exp(
                    dt / self.tau
                )
            )

            synapse.weight -= dw

        # decay phải ở đây
        synapse.weight *= (
            1.0 - self.weight_decay
        )

        synapse.clamp()