from ..core.synapse import Synapse


def connect(
    pre_neuron,
    post_neuron,
    weight=0.5,
    delay=1
):

    synapse = Synapse(
        weight=weight,
        delay=delay,
        pre=pre_neuron,
        post=post_neuron
    )

    pre_neuron.axon.branch(
        synapse
    )

    post_neuron.dendrite.attach(
        synapse
    )

    return synapse