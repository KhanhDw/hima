class Dendrite:
    def __init__(self):
        self.synapses = []

    def attach(self, synapse):
        self.synapses.append(synapse)
        return synapse

    def current(self, time):
        return sum(
            synapse.transmit(time)
            for synapse in self.synapses
        )


    def incoming_synapses(self):
        return self.synapses