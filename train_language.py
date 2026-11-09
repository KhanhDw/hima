"""
train_language.py — Huấn luyện SNN trên bộ dữ liệu ngôn ngữ GreenNode/mtop-intent-vn.

Chiến lược:
  - Mỗi episode sample N mẫu ngẫu nhiên (mặc định 2000) → nhanh hơn nhiều.
  - Warm-up 8 episode đầu: SNN thuần, không arbiter → world model có data.
  - Map 113 intent → 3 output class qua `intent_id % 3`.
  - Báo cáo: accuracy thật, override rate, episodic blend rate, world trust.

Usage:
    python train_language.py
    python train_language.py --episodes 30 --samples 2000 --warmup 8
"""

import argparse
import random

from src.integration.main_cognitive_core import MainCognitiveCore
from data.linguistic_dataset import LinguisticDataset


class LinguisticEnvironment:
    """
    Wrapper biến LinguisticDataset thành environment tương thích MainCognitiveCore.

    - reset() → (vector_0, target_0)
    - step(action) → (vector_next, reward, done)
        reward = +1.0 nếu action == target, ngược lại -1.0
    - target_sequence → list các target (đã map sang 0/1/2)
    - current_step_idx → vị trí hiện tại
    """

    def __init__(self, samples, target_map, state_dim=8):
        self.samples = samples
        self.target_map = target_map
        self.state_dim = state_dim
        self.target_sequence = []
        self.current_step_idx = 0
        # Đảm bảo có target_sequence ngay từ đầu
        self._rebuild_targets()

    def _rebuild_targets(self):
        self.target_sequence = [
            int(self.target_map.get(s["target"], 0))
            for s in self.samples
        ]

    def _pad(self, vec):
        arr = list(vec)[:self.state_dim]
        if len(arr) < self.state_dim:
            arr += [0.0] * (self.state_dim - len(arr))
        return arr

    def reset(self):
        self.current_step_idx = 0
        self._rebuild_targets()
        if not self.samples:
            return [0.0] * self.state_dim, 0
        s0 = self.samples[0]
        return self._pad(s0["vector"]), self.target_sequence[0]

    def step(self, action):
        if self.current_step_idx >= len(self.samples):
            return [0.0] * self.state_dim, 0.0, True

        sample = self.samples[self.current_step_idx]
        target = self.target_sequence[self.current_step_idx]
        reward = 1.0 if action == target else -1.0

        self.current_step_idx += 1
        done = self.current_step_idx >= len(self.samples)

        if done:
            next_vec = [0.0] * self.state_dim
            self.current_step_idx = 0          
        else:
            next_vec = self._pad(
                self.samples[self.current_step_idx]["vector"]
            )

        return next_vec, reward, done


def build_intent_to_3class(all_samples):
    """Map 113 intent_id → 3 class (0/1/2) — workaround cho 3 output neuron."""
    unique_intents = sorted(set(s["target"] for s in all_samples))
    return {
        int(iid): idx % 3
        for idx, iid in enumerate(unique_intents)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--samples", type=int, default=2000,
                        help="Samples per episode")
    parser.add_argument("--warmup", type=int, default=8,
                        help="Warmup episodes (SNN-only, no arbiter)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("HUẤN LUYỆN SNN VỚI BỘ DỮ LIỆU NGÔN NGỮ MTOP-INTENT-VN")
    print("=" * 70)
    print(f"Episodes:        {args.episodes}")
    print(f"Samples/episode: {args.samples}")
    print(f"Warmup episodes: {args.warmup}")
    print(f"Seed:            {args.seed}")
    print("=" * 70)

    # 1. Load dataset
    dataset = LinguisticDataset(split="train")
    all_samples = dataset.dialogue_samples
    print(f"Tổng mẫu có sẵn: {len(all_samples)}")

    # 2. Map intent → 3 class
    intent_to_3class = build_intent_to_3class(all_samples)
    print(f"Map {len(intent_to_3class)} intent → 3 output class")
    print(f"Phân bố 3 class: ", end="")
    dist = {0: 0, 1: 0, 2: 0}
    for s in all_samples:
        dist[intent_to_3class[s["target"]]] += 1
    total = len(all_samples)
    for k, v in dist.items():
        print(f"{k}={v}({v*100/total:.1f}%) ", end="")
    print()

    # 3. Khởi tạo core với warmup
    core = MainCognitiveCore(warmup_episodes=args.warmup)

    # 4. Train loop
    print("\n" + "=" * 70)
    print("BẮT ĐẦU TRAIN")
    print("=" * 70)

    best_acc = 0.0
    for episode in range(args.episodes):
        # Sample subset
        sample_size = min(args.samples, len(all_samples))
        sampled = random.sample(all_samples, sample_size)

        # Swap environment
        core.environment = LinguisticEnvironment(
            sampled, intent_to_3class, state_dim=8
        )

        # Run episode (max_samples=None → dùng hết sampled)
        res = core.run_cognitive_episode(learning=True)

        # Log
        warmup_tag = ""
        if episode < args.warmup:
            warmup_tag = "[WARMUP] "

        print(
            f"Ep {episode+1:02d}/{args.episodes} | {warmup_tag}"
            f"Acc: {res['accuracy']*100:5.1f}% | "
            f"R: {res['reward']:+7.1f} | "
            f"Syn: {res['synapses']:>3} | "
            f"WTr: {res['world_transitions']:>5} | "
            f"WErr: {res['world_error']:.3f} | "
            f"Trust: {res['world_trust']:.2f} | "
            f"Arb: {res['arbiter_override_rate']*100:3.0f}% | "
            f"EpBl: {res['episodic_blend_rate']*100:3.0f}% | "
            f"Epis: {res['episodic_size']:>5}"
        )

        if res["accuracy"] > best_acc:
            best_acc = res["accuracy"]

    # 5. Chat demo
    print("\n" + "=" * 70)
    print(f"Best accuracy: {best_acc*100:.1f}%")
    print("=" * 70)

    print("\n=== Chat mode test ===")
    test_text = "tôi muốn đặt vé máy bay"
    test_vec = dataset.encode_text(test_text)
    status = core.step(test_vec)
    print(f"Input: {test_text}")
    print(f"Cognitive state: {status['state']}")
    print(f"Adaptive LR:     {status['adaptive_learning_rate']:.4f}")
    print(f"Last action:     {core.last_action} (CHAT_ACTION sentinel)")


if __name__ == "__main__":
    main()