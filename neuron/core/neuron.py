from .axon import Axon
from .soma import Soma
from .dendrite import Dendrite


class Neuron:

    def __init__(
        self,
        theta=1.0,
        leak=0.9,
        refractory_period=2,
        rate_alpha=0.01
    ):
        self.axon = Axon()

        self.dendrite = Dendrite()

        self.soma = Soma(
            theta=theta,
            leak=leak,
            refractory_period=refractory_period
        )

        self.rate_alpha = rate_alpha

        self.last_spike_time = -1

        self.fire_count = 0

        self.steps = 0

        self.firing_rate = 0.0

    def step(self, time):

        self.steps += 1

        current = self.dendrite.current(time)

        spike = self.soma.step(
            current
        )

        if spike:

            self.fire_count += 1

            self.last_spike_time = (
                time
            )

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

        return spike

    def reset(self):

        self.soma.reset()

        self.last_spike_time = -1

        self.fire_count = 0

        self.steps = 0

        self.firing_rate = 0.0