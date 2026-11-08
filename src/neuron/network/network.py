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

        # Cập nhật Eligibility Trace cho toàn bộ synapse sau mỗi bước thời gian
        for synapse in self.synapses:
            pre_spiked = getattr(synapse.pre, 'has_spiked', False) if synapse.pre else False
            post_spiked = getattr(synapse.post, 'has_spiked', False) if synapse.post else False
            
            if hasattr(synapse, 'update_trace'):
                synapse.update_trace(pre_spiked, post_spiked)

        return outputs

    def reset(self):

        self.clock.reset()

        for neuron in self.neurons:

            neuron.reset()