from neuron.network.network import Network
from neuron.network.connector import connect
from neuron.core.input_neuron import InputNeuron
from neuron.core.neuron import Neuron
from neuron.learning.trainer import Trainer
from neuron.learning.reward_manager import RewardManager
from neuron.rl.agent import Agent
from neuron.rl.environment import Environment
from neuron.rl.episode_runner import EpisodeRunner

# =========================
# 1. KHỞI TẠO MẠNG & NEURON
# =========================
net = Network()

input_neurons = [
    InputNeuron(),
    InputNeuron(),
    InputNeuron(),
]

out_a = Neuron()
out_b = Neuron()
out_c = Neuron()

for inp in input_neurons:
    net.add(inp)
net.add(out_a)
net.add(out_b)
net.add(out_c)

# =========================
# 2. KẾT NỐI MẠNG BAN ĐẦU
# =========================
net.add_synapse(connect(input_neurons[0], out_a, weight=1.0, delay=1))
net.add_synapse(connect(input_neurons[1], out_b, weight=1.0, delay=1))
net.add_synapse(connect(input_neurons[2], out_c, weight=1.0, delay=1))

# =========================
# 3. THÀNH PHẦN RL & TRAINER
# =========================
trainer = Trainer()
reward_manager = RewardManager(positive_reward=1.0, negative_reward=-1.0)
agent = Agent(output_neurons=[out_a, out_b, out_c])
environment = Environment()

# =========================
# 4. KHỞI TẠO EPISODE RUNNER
# =========================
runner = EpisodeRunner(
    environment=environment,
    network=net,
    input_neurons=input_neurons,
    agent=agent,
    trainer=trainer,
    reward_manager=reward_manager,
)

# =========================
# 5. CHẠY HUẤN LUYỆN & ĐÁNH GIÁ
# =========================
if __name__ == "__main__":
    print(f"Số lượng neuron ban đầu: {len(net.neurons)}")
    print(f"Số lượng synapse ban đầu: {len(net.synapses)}\n")
    print("Bắt đầu huấn luyện phân loại vector...")

    history = runner.train(episodes=150)

    print("\nQuá trình huấn luyện hoàn tất!")
    print(f"Tổng số synapse cuối cùng: {len(net.synapses)}")

    # Chạy chế độ kiểm định phân loại (Evaluation)
    runner.evaluate(num_tests=30)