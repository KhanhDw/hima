class Homeostasis:

    def __init__(
        self,
        target_rate=0.10,
        adapt_rate=0.01
    ):
        self.target_rate = (
            target_rate
        )

        self.adapt_rate = (
            adapt_rate
        )

    def update(
        self,
        neuron
    ):

        error = (
            neuron.firing_rate
            - self.target_rate
        )

        neuron.soma.theta += (
            error
            * self.adapt_rate
        )

        neuron.soma.theta = max(
            0.1,
            min(
                5.0,
                neuron.soma.theta
            )
        )