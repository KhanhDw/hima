from src.integration.main_cognitive_core import MainCognitiveCore
from src.data.linguistic_dataset import LinguisticDataset

def train_snn_on_language(episodes=50):
    print("=== BẮT ĐẦU HUẤN LUYỆN SNN VỚI BỘ DỮ LIỆU NGÔN NGỮ ===")
    core = MainCognitiveCore()
    dataset = LinguisticDataset()
    batch = dataset.get_training_batch()

    for ep in range(episodes):
        total_reward = 0.0
        correct_count = 0

        for item in batch:
            sensory_vec = item["vector"]
            target = item["target"]

            # 1. Đưa vector ngôn ngữ vào vòng lặp nhận thức Cấp 13
            cognitive_res = core.step(sensory_vec)

            # 2. Chạy mạng SNN phần cứng để kích hoạt các neuron đầu ra
            for _ in range(20):
                for i, val in enumerate(sensory_vec):
                    if val > 0.5 and i < len(core.input_neurons):
                        core.input_neurons[i].stimulate(spike=1, time=core.net.clock.time)
                core.net.step()

            action = core.agent.act()
            
            # 3. Đánh giá phần thưởng dựa trên target thực tế
            is_correct = (action == target)
            reward = 1.0 if is_correct else -0.5
            if is_correct:
                correct_count += 1
            total_reward += reward

            # 4. Huấn luyện trọng số Synapse thông qua Trainer với target chuẩn xác
            core.trainer.train_step(core.net, action=action, target=target)
            if core.sub_region_manager:
                core.sub_region_manager.update_region_performance(target, is_correct)

        accuracy = (correct_count / len(batch)) * 100
        if ep % 5 == 0 or ep == episodes - 1:
            print(f"Episode {ep:02d}/{episodes} | Độ chính xác nhận diện: {accuracy:.1f}% | Tổng Reward: {total_reward:+.1f} | Synapses: {len(core.net.synapses)}")

    print("\n=== HOÀN THÀNH HUẤN LUYỆN NGÔN NGỮ ===")
    print(f"Tổng số synapse hình thành: {len(core.net.synapses)}")
    print(f"Tổng số chunks kết tinh: {len(core.chunker.get_chunks())}")

if __name__ == "__main__":
    train_snn_on_language(episodes=30)