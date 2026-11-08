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
        self.activity_threshold = activity_threshold

    def prune(self, neuron):
        before = len(neuron.dendrite.synapses)
        neuron.dendrite.synapses = [
            syn for syn in neuron.dendrite.synapses
            if syn.weight > self.prune_threshold
        ]
        return before - len(neuron.dendrite.synapses)

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

    def maybe_grow(
        self,
        pre,
        post,
        delay=1
    , input_neurons=None, output_neurons=None):
        """
        Mọc synapse thông minh theo tiểu vùng (Sub-region / Modular Growth):
        Chỉ cho phép mọc nếu cặp (pre, post) nằm trong cùng một tiểu vùng logic 
        (ví dụ: Input thứ i đi chung với Output thứ i) để tránh chéo kênh.
        """
        if pre is post:
            return None

        if self.connected(pre, post):
            return None

        # Nếu có định nghĩa danh sách tiểu vùng, ta giới hạn phạm vi mọc trong cụm độc lập
        if input_neurons and output_neurons:
            if pre in input_neurons and post in output_neurons:
                # Kiểm tra đúng index tương ứng trong tiểu vùng (ví dụ: input[0] chỉ mọc vào output[0])
                try:
                    idx_pre = input_neurons.index(pre)
                    idx_post = output_neurons.index(post)
                    if idx_pre != idx_post:
                        return None # Không cho phép mọc chéo tiểu vùng khác
                except ValueError:
                    pass

        if (
            pre.firing_rate > 0.15 and  
            post.firing_rate > 0.15
        ):
            return self.grow(pre, post, delay)

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