from neuron.plasticity.homeostasis import Homeostasis
from neuron.plasticity.structural import StructuralPlasticity
from neuron.learning.plasticity_manager import PlasticityManager
from neuron.learning.reward_manager import RewardManager

class Trainer:

    def __init__(self, reward_manager=None):
        self.homeostasis = Homeostasis()
        self.structural = StructuralPlasticity()
        self.plasticity = PlasticityManager()
        self.reward_manager = reward_manager if reward_manager else RewardManager()

    def update_network(self, network):
        # 1. Cập nhật Homeostasis và Structural Plasticity cho từng neuron
        for neuron in network.neurons:
            self.homeostasis.update(neuron)
            self.structural.prune(neuron)
            self.plasticity.update_neuron(neuron)

        # 2. Xem xét mọc synapse mới giữa các neuron trong mạng
        # neurons = network.neurons
        # for pre in neurons:
        #     for post in neurons:
        #         if pre is post:
        #             continue

        #         synapse = self.structural.maybe_grow(pre, post)
        #         if synapse:
        #             network.add_synapse(synapse)

    def train_step(self, network, action, target):
        """
        Thực hiện bước cập nhật trọng số dựa trên phần thưởng RL (R-STDP)
        """
        # Cập nhật cấu trúc mạng cơ bản
        self.update_network(network)

        # Đánh giá và áp dụng reward-modulated STDP
        if action is not None and target is not None:
            self.reward_manager.evaluate(network, action, target)