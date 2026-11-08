from .neuron import Neuron


class InputNeuron(Neuron):

    def stimulate(
        self,
        spike,
        time
    ):

        if not spike:
            return

        self.steps += 1

        self.fire_count += 1

        self.last_spike_time = time

        self.axon.transmit(
            spike,
            time
        )

        self.firing_rate = (
            (1 - self.rate_alpha)
            * self.firing_rate
            +
            self.rate_alpha
            * spike
        )