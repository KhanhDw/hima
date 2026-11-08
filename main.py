from src.neuron.network.network import Network
from src.neuron.network.connector import connect
from src.neuron.core.input_neuron import InputNeuron
from src.neuron.core.neuron import Neuron
from src.neuron.learning.trainer import Trainer
from src.neuron.learning.reward_manager import RewardManager
from src.neuron.learning.sub_region_manager import SubRegionManager
from src.neuron.rl.agent import Agent
from src.neuron.rl.environment import Environment

# Import các module nhận thức mới (Representation, Memory, Prediction)
from src.representation.encoder import Encoder
from src.memory.sequence_memory import SequenceMemory
from src.memory.context_memory import ContextMemory
from src.memory.working_memory import WorkingMemory
from src.prediction.predictor import Predictor
from src.prediction.prediction_error import PredictionError
from src.prediction.confidence import ConfidenceEstimator
from src.prediction.surprise import SurpriseCalculator

# =========================
# 1. KHỞI TẠO MẠNG & INPUT NEURONS
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

# Kết nối mạng cơ bản
net.add_synapse(connect(input_neurons[0], out_a, weight=1.0, delay=1))
net.add_synapse(connect(input_neurons[1], out_b, weight=1.0, delay=1))
net.add_synapse(connect(input_neurons[2], out_c, weight=1.0, delay=1))

for i in range(3, 8):
    net.add_synapse(connect(input_neurons[i], out_a, weight=0.2, delay=1))
    net.add_synapse(connect(input_neurons[i], out_b, weight=0.2, delay=1))
    net.add_synapse(connect(input_neurons[i], out_c, weight=0.2, delay=1))

# =========================
# 2. KHỞI TẠO HỆ THỐNG NHẬN THỨC (CẤP 3, 4, 5)
# =========================
encoder = Encoder(num_neurons=8, min_val=0.0, max_val=2.0)
sequence_memory = SequenceMemory(max_history_length=10)
context_memory = ContextMemory()
working_memory = WorkingMemory(capacity=5)

predictor = Predictor()

# =========================
# 3. QUẢN LÝ TIỂU VÙNG & TRAINER
# =========================
sub_region_manager = SubRegionManager(freeze_threshold=90.0)
sub_region_manager.register_region("Task_0", input_indices=[0, 3, 7], output_index=0)
sub_region_manager.register_region("Task_1", input_indices=[1, 4, 6], output_index=1)
sub_region_manager.register_region("Task_2", input_indices=[2, 5], output_index=2)

trainer = Trainer(
    input_neurons=input_neurons,
    output_neurons=[out_a, out_b, out_c],
    sub_region_manager=sub_region_manager
)
reward_manager = RewardManager(positive_reward=1.0, negative_reward=-1.0)
agent = Agent(output_neurons=[out_a, out_b, out_c])
environment = Environment()


# =========================
# 4. VÒNG LẶP HUẤN LUYỆN TỰ GIÁM SÁT & DỰ ĐOÁN (PREDICTIVE LEARNING LOOP)
# =========================
def run_predictive_episode(environment, network, input_neurons, agent, trainer, encoder, seq_memory, work_memory, predictor, learning=True):
    state_vector, target = environment.reset()
    done = False
    total_reward = 0.0
    steps_count = 0
    
    # Khởi tạo giá trị mặc định để tránh lỗi unbound variable
    error = 0.0
    surprise = 0.0
    confidence = 1.0
    
    max_steps = len(environment.target_sequence)

    while not done and steps_count < max_steps:
        steps_count += 1

        # 1. Biểu diễn trạng thái qua Encoder (Cấp 3)
        latent = encoder.encode_state(state_vector)

        # 2. Truy xuất bộ nhớ chuỗi & làm việc (Cấp 4)
        seq_memory.append(latent)
        work_memory.load(latent)

        # 3. Thực hiện dự đoán bước tiếp theo (Cấp 5)
        history = seq_memory.get_sequence()
        predicted_latent = predictor.predict(history)

        # Kích hoạt nơ-ron qua SNN như cũ
        for _ in range(30):
            for i, val in enumerate(state_vector):
                if val == 1 and i < len(input_neurons):
                    input_neurons[i].stimulate(spike=1, time=network.clock.time)
            network.step()

        action = agent.act()
        next_state_vector, reward, done = environment.step(action)
        total_reward += reward

        # 4. Tính toán sai số dự đoán (Prediction Error & Surprise) nếu có bước tiếp theo
        next_latent = encoder.encode_state(next_state_vector)
        error = PredictionError.calculate(predicted_latent, next_latent)
        surprise = SurpriseCalculator.calculate(error)
        confidence = ConfidenceEstimator.estimate(error)

        # 5. Huấn luyện (Kết hợp RL reward cũ với tín hiệu tự giám sát từ sai số dự đoán)
        is_step_correct = (reward > 0)
        if learning:
            trainer.train_step(network, action, target)
            if sub_region_manager:
                sub_region_manager.update_region_performance(target, is_step_correct)

        target = environment.target_sequence[environment.current_step_idx]
        state_vector = next_state_vector

    is_correct = (total_reward >= len(environment.target_sequence))
    return {
        "correct": is_correct,
        "reward": total_reward,
        "synapses": len(network.synapses),
        "last_error": error,
        "surprise": surprise,
        "confidence": confidence
    }


if __name__ == "__main__":
    print(f"Số lượng neuron ban đầu: {len(net.neurons)}")
    print(f"Số lượng synapse ban đầu: {len(net.synapses)}\n")
    print("Bắt đầu huấn luyện chuỗi tích hợp Predictive Coding (Cấp 3 -> Cấp 5)...")

    recent_correct = []
    for episode in range(150):
        res = run_predictive_episode(
            environment, net, input_neurons, agent, trainer,
            encoder, sequence_memory, working_memory, predictor, learning=True
        )
        recent_correct.append(1 if res["correct"] else 0)
        if len(recent_correct) > 20:
            recent_correct.pop(0)

        if episode % 10 == 0:
            acc = sum(recent_correct) / len(recent_correct) * 100
            print(
                f"episode={episode:02d} | "
                f"reward={res['reward']:+.1f} | "
                f"acc_20={acc:.1f}% | "
                f"error={res['last_error']:.4f} | "
                f"surprise={res['surprise']:.4f} | "
                f"confidence={res['confidence']:.4f} | "
                f"syn={res['synapses']}"
            )

    print("\nQuá trình huấn luyện Predictive Coding hoàn tất!")
    print(f"Tổng số synapse cuối cùng: {len(net.synapses)}")