from neuron.network.network import Network
from neuron.network.connector import connect
from neuron.core.input_neuron import InputNeuron
from neuron.core.neuron import Neuron
from neuron.learning.trainer import Trainer
from neuron.learning.reward_manager import RewardManager
from neuron.learning.sub_region_manager import SubRegionManager
from neuron.rl.agent import Agent
from neuron.rl.environment import Environment
from neuron.rl.episode_runner import EpisodeRunner

# =========================
# 1. KHỞI TẠO MẠNG & 8 INPUT NEURONS (3 GIÁ TRỊ + 5 VỊ TRÍ BƯỚC)
# =========================
net = Network()

input_neurons = [InputNeuron() for _ in range(8)]

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
# Kết nối từ 3 input giá trị cơ bản đến các output neuron tương ứng
net.add_synapse(connect(input_neurons[0], out_a, weight=1.0, delay=1))
net.add_synapse(connect(input_neurons[1], out_b, weight=1.0, delay=1))
net.add_synapse(connect(input_neurons[2], out_c, weight=1.0, delay=1))

# Kết nối bổ trợ từ các input vị trí bước để giúp mạng nhận biết ngữ cảnh thời gian
for i in range(3, 8):
    net.add_synapse(connect(input_neurons[i], out_a, weight=0.2, delay=1))
    net.add_synapse(connect(input_neurons[i], out_b, weight=0.2, delay=1))
    net.add_synapse(connect(input_neurons[i], out_c, weight=0.2, delay=1))

# =========================
# 3. KHỞI TẠO QUẢN LÝ TIỂU VÙNG (SUB-REGION MANAGER)
# =========================
sub_region_manager = SubRegionManager(freeze_threshold=90.0)
sub_region_manager.register_region("Task_0", input_indices=[0, 3, 7], output_index=0)
sub_region_manager.register_region("Task_1", input_indices=[1, 4, 6], output_index=1)
sub_region_manager.register_region("Task_2", input_indices=[2, 5], output_index=2)

# =========================
# 4. THÀNH PHẦN RL & TRAINER
# =========================
trainer = Trainer(
    input_neurons=list(input_neurons.values()) if isinstance(input_neurons, dict) else input_neurons,
    output_neurons=[out_a, out_b, out_c],
    sub_region_manager=sub_region_manager
)
reward_manager = RewardManager(positive_reward=1.0, negative_reward=-1.0)
agent = Agent(output_neurons=[out_a, out_b, out_c])
environment = Environment()

# =========================
# 5. KHỞI TẠO EPISODE RUNNER
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
# 6. CHẠY HUẤN LUYỆN & ĐÁNH GIÁ
# =========================
if __name__ == "__main__":
    print(f"Số lượng neuron ban đầu: {len(net.neurons)}")
    print(f"Số lượng synapse ban đầu: {len(net.synapses)}\n")
    print("Bắt đầu huấn luyện chuỗi 5 bước lặp lại với Temporal Encoding...")

    history = runner.train(episodes=300)

    print("\nQuá trình huấn luyện hoàn tất!")
    print(f"Tổng số synapse cuối cùng: {len(net.synapses)}")

    runner.evaluate(num_tests=30)