class Axon:
    def __init__(self):
        self.synapses = []

    def branch(self, synapse):
        self.synapses.append(synapse)
        return synapse

    def transmit(self, spike, time):

        for synapse in self.synapses:

            synapse.receive(
                spike=spike,
                time=time
            )