from src.neuron.network.network import Network
from src.neuron.network.connector import connect
from src.neuron.core.input_neuron import InputNeuron
from src.neuron.core.neuron import Neuron
from src.neuron.learning.trainer import Trainer
from src.neuron.learning.reward_manager import RewardManager
from src.neuron.learning.sub_region_manager import SubRegionManager
from src.neuron.rl.agent import Agent
from src.neuron.rl.environment import Environment

# 1. Module Biểu diễn & Bộ nhớ (Cấp 3 & 4)
from src.representation.encoder import Encoder
from src.memory.sequence_memory import SequenceMemory
from src.memory.context_memory import ContextMemory
from src.memory.working_memory import WorkingMemory

# 2. Module Dự đoán (Cấp 5)
from src.prediction.predictor import Predictor
from src.prediction.prediction_error import PredictionError
from src.prediction.confidence import ConfidenceEstimator
from src.prediction.surprise import SurpriseCalculator

# 3. Module Trừu tượng hóa Phân cấp (Cấp 6)
from src.hierarchy.chunker import Chunker
from src.hierarchy.abstraction_layer import AbstractionLayer
from src.hierarchy.meta_representation import MetaRepresentation

# 4. Module Mục tiêu & Lập kế hoạch (Cấp 7)
from src.planning.goal_manager import GoalManager
from src.planning.planner import Planner
from src.planning.intrinsic_reward import IntrinsicRewardCalculator


# =========================
# 1. KHỞI TẠO MẠNG SNN NỀN TẢNG
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

# Kết nối cơ bản
net.add_synapse(connect(input_neurons[0], out_a, weight=1.0, delay=1))
net.add_synapse(connect(input_neurons[1], out_b, weight=1.0, delay=1))
net.add_synapse(connect(input_neurons[2], out_c, weight=1.0, delay=1))

for i in range(3, 8):
    net.add_synapse(connect(input_neurons[i], out_a, weight=0.2, delay=1))
    net.add_synapse(connect(input_neurons[i], out_b, weight=0.2, delay=1))
    net.add_synapse(connect(input_neurons[i], out_c, weight=0.2, delay=1))

# =========================
# 2. KHỞI TẠO CÁC PHÂN HỆ NHẬN THỨC (CẤP 3 - CẤP 7)
# =========================
encoder = Encoder(num_neurons=8, min_val=0.0, max_val=2.0)
sequence_memory = SequenceMemory(max_history_length=10)
context_memory = ContextMemory()
working_memory = WorkingMemory(capacity=5)

predictor = Predictor()

chunker = Chunker(frequency_threshold=3)
abstraction_layer = AbstractionLayer()
meta_rep = MetaRepresentation()

goal_manager = GoalManager()
planner = Planner()

# =========================
# 3. QUẢN LÝ TIỂU VÙNG & MÔ TRƯỜNG
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
# 4. VÒNG LẶP HUẤN LUYỆN TỔNG HỢP (LEVEL 1 -> LEVEL 7)
# =========================
def run_full_cognitive_episode(learning=True):
    state_vector, target = environment.reset()
    done = False
    total_reward = 0.0
    steps_count = 0

    error = 0.0
    surprise = 0.0
    confidence = 1.0
    intrinsic_reward = 0.0
    active_chunk_id = None

    # Đặt mục tiêu (Goal): Trạng thái hoàn thành chuỗi mục tiêu
    target_goal_vector = [1, 0, 0, 0, 0, 1, 0, 0, 1] # Trạng thái đích mong muốn
    goal_manager.set_goal(target_goal_vector)

    max_steps = len(environment.target_sequence)

    while not done and steps_count < max_steps:
        steps_count += 1

        # A. Representation (Cấp 3)
        latent = encoder.encode_state(state_vector)

        # B. Memory (Cấp 4)
        sequence_memory.append(latent)
        working_memory.load(latent)

        # C. Hierarchical Abstraction (Cấp 6)
        # Theo dõi chuỗi lịch sử để phát hiện khái niệm/chunk lặp lại
        history = sequence_memory.get_sequence()
        if len(history) >= 2:
            active_chunk_id = chunker.observe_sequence(history[-2:])
            if active_chunk_id:
                abstraction_layer.elevate(history[-1], active_chunk_id)
                meta_rep.link("HighLevelConcept", active_chunk_id)

        # D. Goal & Planning (Cấp 7)
        # D. Goal & Planning (Cấp 7)
        current_goal = goal_manager.get_goal() or []
        plan = planner.generate_plan(latent, current_goal)
        prev_dist = sum(abs(g - s) for g, s in zip(current_goal, state_vector)) if current_goal else 1.0

        # E. Prediction (Cấp 5)
        predicted_latent = predictor.predict(history)

        # F. Kích hoạt mạng SNN
        for _ in range(30):
            for i, val in enumerate(state_vector):
                if val == 1 and i < len(input_neurons):
                    input_neurons[i].stimulate(spike=1, time=net.clock.time)
            net.step()

        action = agent.act()
        next_state_vector, ext_reward, done = environment.step(action)
        
        # G. Intrinsic Reward & Goal Check (Cấp 7)
        curr_dist = sum(abs(g - s) for g, s in zip(current_goal, next_state_vector)) if current_goal else 1.0
        intrinsic_reward = IntrinsicRewardCalculator.calculate(prev_dist, curr_dist)
        goal_manager.check_achieved(next_state_vector)

        # H. Prediction Error & Surprise (Cấp 5)
        next_latent = encoder.encode_state(next_state_vector)
        error = PredictionError.calculate(predicted_latent, next_latent)
        surprise = SurpriseCalculator.calculate(error)
        confidence = ConfidenceEstimator.estimate(error)

        # I. Tổng hợp phần thưởng & Huấn luyện SNN
        combined_reward = ext_reward + (0.1 * intrinsic_reward)
        total_reward += combined_reward

        is_step_correct = (ext_reward > 0)
        if learning:
            trainer.train_step(net, action, target)
            if sub_region_manager:
                sub_region_manager.update_region_performance(target, is_step_correct)

        target = environment.target_sequence[environment.current_step_idx]
        state_vector = next_state_vector

    is_correct = (total_reward >= len(environment.target_sequence))
    return {
        "correct": is_correct,
        "reward": total_reward,
        "synapses": len(net.synapses),
        "last_error": error,
        "surprise": surprise,
        "confidence": confidence,
        "intrinsic_reward": intrinsic_reward,
        "chunk_count": len(chunker.get_chunks()),
        "goal_achieved": goal_manager.is_achieved
    }


# =========================
# 5. CHẠY THỬ NGHIỆM TÍCH HỢP TOÀN HỆ THỐNG
# =========================
if __name__ == "__main__":
    print(f"Số lượng neuron ban đầu: {len(net.neurons)}")
    print(f"Số lượng synapse ban đầu: {len(net.synapses)}\n")
    print("Bắt đầu huấn luyện hệ thống SNN tích hợp toàn diện (Cấp 1 -> Cấp 7)...")

    recent_correct = []
    for episode in range(150):
        res = run_full_cognitive_episode(learning=True)
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
                f"chunks={res['chunk_count']} | "
                f"goal_achieved={res['goal_achieved']} | "
                f"syn={res['synapses']}"
            )

    print("\nQuá trình huấn luyện tích hợp toàn diện Cấp 1-7 hoàn tất!")
    print(f"Tổng số chunk khái niệm được kết tinh: {len(chunker.get_chunks())}")
    print(f"Tổng số synapse cuối cùng: {len(net.synapses)}")