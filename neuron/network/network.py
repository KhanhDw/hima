from ..core.clock import Clock


class Network:

    def __init__(self):

        self.clock = Clock()
        self.neurons = []
        self.synapses = []

    def add(self, neuron):

        self.neurons.append(neuron)

        return neuron
    
    def add_synapse(
        self,
        synapse
    ):
        self.synapses.append(
            synapse
        )

        return synapse


    def step(self):

        time = self.clock.tick()

        outputs = []

        for neuron in self.neurons:

            outputs.append(
                neuron.step(time)
            )

        return outputs

    def reset(self):

        self.clock.reset()

        for neuron in self.neurons:

            neuron.reset()