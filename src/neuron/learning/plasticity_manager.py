from ..plasticity.stdp import STDP
from ..plasticity.homeostasis import Homeostasis


class PlasticityManager:

    def __init__(self):

        self.stdp = STDP()

        self.homeostasis = (
            Homeostasis()
        )

    def update_neuron(
        self,
        neuron
    ):

        self.homeostasis.update(
            neuron
        )

        if neuron.last_spike_time < 0:
            return

        for synapse in (
            neuron.dendrite.incoming_synapses()
        ):

            self.stdp.update(
                synapse,
                neuron.last_spike_time
            )