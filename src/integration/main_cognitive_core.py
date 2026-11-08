from src.integration.global_workspace import GlobalWorkspace
from src.integration.cognitive_orchestrator import CognitiveOrchestrator
from src.autonomous_goal.drive_motivator import DriveMotivator
from src.autonomous_goal.curiosity_engine import CuriosityEngine
from src.autonomous_goal.goal_generator import GoalGenerator
from src.meta.meta_learner import MetaLearner
from src.meta.performance_monitor import PerformanceMonitor

# Import các tầng nền tảng (Cấp 1 - 7)
from src.neuron.network.network import Network
from src.neuron.core.input_neuron import InputNeuron
from src.neuron.core.neuron import Neuron
from src.neuron.learning.trainer import Trainer
from src.neuron.learning.reward_manager import RewardManager
from src.neuron.learning.sub_region_manager import SubRegionManager
from src.neuron.rl.agent import Agent
from src.neuron.rl.environment import Environment

from src.representation.encoder import Encoder
from src.memory.sequence_memory import SequenceMemory
from src.memory.context_memory import ContextMemory
from src.memory.working_memory import WorkingMemory

from src.prediction.predictor import Predictor
from src.prediction.prediction_error import PredictionError
from src.prediction.confidence import ConfidenceEstimator
from src.prediction.surprise import SurpriseCalculator

from src.hierarchy.chunker import Chunker
from src.hierarchy.abstraction_layer import AbstractionLayer
from src.hierarchy.meta_representation import MetaRepresentation

from src.planning.goal_manager import GoalManager
from src.planning.planner import Planner
from src.planning.intrinsic_reward import IntrinsicRewardCalculator

class MainCognitiveCore:
    def __init__(self):
        """
        Trái tim nhận thức toàn cục (Main Cognitive Core) tích hợp toàn bộ 13 cấp độ 
        của hệ thống SNN thành một thực thể tự chủ hoàn chỉnh, hợp nhất phần cứng SNN
        với các tầng nhận thức cao cấp.
        """
        print("Đang khởi tạo Tích hợp Nhận thức Toàn cục (Cấp 13) & Phần cứng SNN...")
        
        # 1. Khởi tạo mạng SNN & Neuron đầu vào/đầu ra (Cấp 1-2)
        self.net = Network()
        self.input_neurons = [InputNeuron() for _ in range(8)]
        self.out_a = Neuron()
        self.out_b = Neuron()
        self.out_c = Neuron()

        for inp in self.input_neurons:
            self.net.add(inp)
        self.net.add(self.out_a)
        self.net.add(self.out_b)
        self.net.add(self.out_c)

        # 2. Khởi tạo các phân hệ nhận thức cơ sở (Cấp 3 - 7)
        self.encoder = Encoder(num_neurons=8, min_val=0.0, max_val=2.0)
        self.sequence_memory = SequenceMemory(max_history_length=10)
        self.context_memory = ContextMemory()
        self.working_memory = WorkingMemory(capacity=5)
        self.predictor = Predictor()
        self.chunker = Chunker(frequency_threshold=3)
        self.abstraction_layer = AbstractionLayer()
        self.meta_rep = MetaRepresentation()
        self.goal_manager = GoalManager()
        self.planner = Planner()

        # Quản lý tiểu vùng & Huấn luyện viên
        self.sub_region_manager = SubRegionManager(freeze_threshold=90.0)
        self.trainer = Trainer(
            input_neurons=self.input_neurons,
            output_neurons=[self.out_a, self.out_b, self.out_c],
            sub_region_manager=self.sub_region_manager
        )
        self.reward_manager = RewardManager(positive_reward=1.0, negative_reward=-1.0)
        self.agent = Agent(output_neurons=[self.out_a, self.out_b, self.out_c])
        self.environment = Environment()

        # 3. Không gian làm việc và Nhạc trưởng (Cấp 13)
        self.workspace = GlobalWorkspace()
        self.orchestrator = CognitiveOrchestrator(self.workspace)
        
        # 4. Động lực, Tự sinh mục tiêu & Siêu học tập (Cấp 10 & 12)
        self.drive_motivator = DriveMotivator()
        self.curiosity_engine = CuriosityEngine()
        self.goal_generator = GoalGenerator()
        self.meta_learner = MetaLearner()
        self.performance_monitor = PerformanceMonitor()
        
        print("Hệ thống SNN Toàn cục đã sẵn sàng vận hành toàn diện từ Cấp 1 đến Cấp 13!")

    def run_cognitive_episode(self, learning=True):
        """
        Thực hiện một episode hoàn chỉnh tích hợp toàn bộ chuỗi nhận thức và vòng lặp SNN.
        """
        state_vector, target = self.environment.reset()
        done = False
        total_reward = 0.0
        steps_count = 0
        max_steps = len(self.environment.target_sequence)

        error = 0.0
        surprise = 0.0
        confidence = 1.0
        cognitive_status = {"state": "IDLE"}

        while not done and steps_count < max_steps:
            steps_count += 1

            # A. Mã hóa biểu diễn (Cấp 3) & Bộ nhớ (Cấp 4)
            latent = self.encoder.encode_state(state_vector)
            self.sequence_memory.append(latent)
            self.working_memory.load(latent)

            # Phát sóng thông tin qua Không gian làm việc toàn cục (Cấp 13)
            conscious_packet = self.workspace.broadcast(latent, source_module="MemoryCore")

            # B. Trừu tượng hóa phân cấp & Chunking (Cấp 6)
            history = self.sequence_memory.get_sequence()
            if len(history) >= 2:
                active_chunk_id = self.chunker.observe_sequence(history[-2:])
                if active_chunk_id:
                    self.abstraction_layer.elevate(history[-1], active_chunk_id)
                    self.meta_rep.link("HighLevelConcept", active_chunk_id)

            # C. Động lực, Tò mò & Tự sinh mục tiêu (Cấp 12)
            drives = self.drive_motivator.update_drives(activity_cost=0.02)
            curiosity_score = self.curiosity_engine.calculate_curiosity(state_vector)
            
            new_goal = self.goal_generator.evaluate_and_generate(curiosity_score, state_vector)
            if new_goal:
                self.goal_manager.set_goal([1, 0, 0, 0, 0, 1, 0, 0, 1])

            # D. Dự đoán tương lai (Cấp 5)
            predicted_latent = self.predictor.predict(history)

            # E. Kích hoạt mạng SNN phần cứng
            for _ in range(30):
                for i, val in enumerate(state_vector):
                    if val == 1 and i < len(self.input_neurons):
                        self.input_neurons[i].stimulate(spike=1, time=self.net.clock.time)
                self.net.step()

            action = self.agent.act()
            next_state_vector, ext_reward, done = self.environment.step(action)
            
            # F. Đánh giá sai số dự đoán & Nhạc trưởng điều phối (Cấp 5 & 13)
            next_latent = self.encoder.encode_state(next_state_vector)
            error = PredictionError.calculate(predicted_latent, next_latent)
            surprise = SurpriseCalculator.calculate(error)
            confidence = ConfidenceEstimator.estimate(error)

            cognitive_status = self.orchestrator.tick(state_vector, drives, curiosity_score)

            # G. Siêu học tập tinh chỉnh tham số (Cấp 10)
            current_lr = self.meta_learner.adapt(recent_accuracy=1.0 if ext_reward > 0 else 0.0)

            # H. Huấn luyện SNN qua Trainer
            combined_reward = ext_reward
            total_reward += combined_reward
            is_step_correct = (ext_reward > 0)

            if learning:
                self.trainer.train_step(self.net, action, target)
                if self.sub_region_manager:
                    self.sub_region_manager.update_region_performance(target, is_step_correct)

            # Ghi nhận hiệu suất
            self.performance_monitor.record_metrics(accuracy=1.0 if is_step_correct else 0.0, error=error, reward=total_reward)

            target = self.environment.target_sequence[self.environment.current_step_idx]
            state_vector = next_state_vector

        is_correct = (total_reward >= len(self.environment.target_sequence))
        return {
            "correct": is_correct,
            "reward": total_reward,
            "synapses": len(self.net.synapses),
            "last_error": error,
            "chunk_count": len(self.chunker.get_chunks()),
            "cognitive_state": cognitive_status["state"]
        }

if __name__ == "__main__":
    core = MainCognitiveCore()
    print("\nBắt đầu chạy thử nghiệm vòng lặp nhận thức toàn cục Cấp 13...")
    
    for episode in range(20):
        res = core.run_cognitive_episode(learning=True)
        print(
            f"Episode {episode:02d} | "
            f"State: {res['cognitive_state']} | "
            f"Reward: {res['reward']:+.1f} | "
            f"Error: {res['last_error']:.4f} | "
            f"Chunks: {res['chunk_count']} | "
            f"Synapses: {res['synapses']}"
        )