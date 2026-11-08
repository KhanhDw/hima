from neuron.plasticity.homeostasis import Homeostasis
from neuron.plasticity.structural import StructuralPlasticity
from neuron.learning.plasticity_manager import PlasticityManager
from neuron.learning.reward_manager import RewardManager

class Trainer:

    def __init__(self, reward_manager=None, input_neurons=None, output_neurons=None, sub_region_manager=None):
        self.homeostasis = Homeostasis()
        self.structural = StructuralPlasticity()
        self.plasticity = PlasticityManager()
        self.reward_manager = reward_manager if reward_manager else RewardManager()
        
        self.input_neurons = input_neurons if input_neurons else []
        self.output_neurons = output_neurons if output_neurons else []
        self.sub_region_manager = sub_region_manager

    def update_network(self, network, target=None):
        # Nếu tiểu vùng chuyên trách target đã đóng băng, bỏ qua cập nhật để bảo vệ biểu diễn
        if self.sub_region_manager and target is not None and self.sub_region_manager.is_target_frozen(target):
            return

        for neuron in network.neurons:
            self.homeostasis.update(neuron)
            self.structural.prune(neuron)
            self.plasticity.update_neuron(neuron)

        neurons = network.neurons
        for pre in neurons:
            for post in neurons:
                if pre is post:
                    continue

                synapse = self.structural.maybe_grow(
                    pre, 
                    post, 
                    delay=1, 
                    input_neurons=self.input_neurons, 
                    output_neurons=self.output_neurons
                )
                if synapse:
                    network.add_synapse(synapse)

    def train_step(self, network, action, target):
        # Cập nhật cấu trúc mạng theo tiểu vùng
        self.update_network(network, target=target)

        # Nếu tiểu vùng đã đóng băng, khóa luôn việc thay đổi trọng số
        if self.sub_region_manager and target is not None and self.sub_region_manager.is_target_frozen(target):
            return

        if action is not None and target is not None:
            # Đánh giá phần thưởng từ RewardManager
            reward = self.reward_manager.evaluate(network, action, target)

            # Lan truyền phần thưởng thông qua Eligibility Trace của từng synapse
            for synapse in network.synapses:
                if hasattr(synapse, 'apply_reward_modulation'):
                    synapse.apply_reward_modulation(reward)