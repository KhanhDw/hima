class Agent:

    def __init__(
        self,
        output_neurons
    ):
        self.output_neurons = (
            output_neurons
        )

    def act(self):

        activities = [

            neuron.firing_rate

            for neuron

            in self.output_neurons
        ]

        if not activities:
            return None

        action = max(
            range(
                len(
                    activities
                )
            ),
            key=lambda i:
            activities[i]
        )

        return action

    def confidence(self):

        activities = [

            neuron.firing_rate

            for neuron

            in self.output_neurons
        ]

        if not activities:
            return 0.0

        return max(
            activities
        )