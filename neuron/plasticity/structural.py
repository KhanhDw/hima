from ..network.connector import connect


class StructuralPlasticity:

    def __init__(
        self,
        prune_threshold=0.05,
        growth_weight=0.10,
        activity_threshold=0.20
    ):
        self.prune_threshold = prune_threshold
        self.growth_weight = growth_weight
        self.activity_threshold = (
            activity_threshold
        )

    def prune(self, neuron):

        before = len(
            neuron.dendrite.synapses
        )

        neuron.dendrite.synapses = [

            syn

            for syn
            in neuron.dendrite.synapses

            if syn.weight >
            self.prune_threshold
        ]

        return (
            before -
            len(neuron.dendrite.synapses)
        )

    def grow(
        self,
        pre_neuron,
        post_neuron,
        delay=1
    ):

        return connect(
            pre_neuron,
            post_neuron,
            weight=self.growth_weight,
            delay=delay
        )

    def should_connect(
        self,
        pre_neuron,
        post_neuron
    ):

        return (

            pre_neuron.firing_rate
            >
            self.activity_threshold

            and

            post_neuron.firing_rate
            >
            self.activity_threshold
        )


    def maybe_grow(
        self,
        pre,
        post,
        delay=1
    ):

        if pre is post:
            return None

        if self.connected(
            pre,
            post
        ):
            return None

        if (
            pre.firing_rate > 0.15
            and
            post.firing_rate > 0.15
        ):

            return self.grow(
                pre,
                post,
                delay
            )

        return None

    def connected(
        self,
        pre,
        post
    ):

        for syn in pre.axon.synapses:

            if syn.post is post:
                return True

        return False