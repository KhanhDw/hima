from ..network.connector import connect


class StructuralPlasticity:

    def __init__(
        self,
        prune_threshold=0.05,
        growth_weight=0.10,
        activity_threshold=0.60,  # Nâng ngưỡng kích hoạt lên cao để hạn chế mọc bừa bãi
        max_synapses=35           # Đặt trần giới hạn tối đa số lượng synapse trong mạng
    ):
        self.prune_threshold = prune_threshold
        self.growth_weight = growth_weight
        self.activity_threshold = activity_threshold
        self.max_synapses = max_synapses

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
        delay=1,
        input_neurons=None,
        output_neurons=None,
        current_synapse_count=0
    ):
        """
        Mọc synapse thông minh theo tiểu vùng (Sub-region / Modular Growth) 
        kèm theo kiểm soát giới hạn trần số lượng synapse.
        """
        # Nếu tổng số synapse đã đạt giới hạn trần, tuyệt đối không mọc thêm
        if current_synapse_count >= self.max_synapses:
            return None

        if pre is post:
            return None

        if self.connected(pre, post):
            return None

        # Nếu có định nghĩa danh sách tiểu vùng, giới hạn phạm vi mọc trong cụm độc lập
        if input_neurons and output_neurons:
            if pre in input_neurons and post in output_neurons:
                try:
                    idx_pre = input_neurons.index(pre)
                    idx_post = output_neurons.index(post)
                    if idx_pre != idx_post:
                        return None 
                except ValueError:
                    pass

        # Chỉ cho phép mọc khi tần số hoạt động vượt qua ngưỡng cao mới
        if (
            pre.firing_rate > self.activity_threshold and  
            post.firing_rate > self.activity_threshold
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