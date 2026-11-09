# src/data/linguistic_dataset.py
import torch
import numpy as np
from datasets import load_dataset
from collections import Counter
import random

class LinguisticDataset:
    def __init__(self, split="train", bow_dim=8, random_seed=42):
        """
        Bộ dữ liệu ngôn ngữ cấu trúc dành cho hệ thống Spiking Neural Network (SNN).
        Sử dụng dữ liệu thật từ GreenNode/mtop-intent-vn.

        Args:
            split (str): Tập dữ liệu để tải ('train', 'validation', 'test').
            bow_dim (int): Số chiều của vector đầu vào SNN (mặc định là 8).
            random_seed (int): Seed để tạo ma trận chiếu cố định.
        """
        self.split = split
        self.bow_dim = bow_dim
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 1. Tải dataset MTOP Intent tiếng Việt
        print(f"Đang tải GreenNode/mtop-intent-vn split='{split}'...")
        raw_dataset = load_dataset(
            "GreenNode/mtop-intent-vn",
            split=split
        )
        
        # 2. Xây dựng từ vựng và ánh xạ intent
        self._build_vocab_and_intent_map(raw_dataset)
        
        # 3. Tạo ma trận chiếu cố định từ không gian BoW sang bow_dim
        self.projection_matrix = np.random.randn(len(self.vocab), self.bow_dim).astype(np.float32)
        
        # 4. Chuyển đổi toàn bộ dataset sang định dạng của LinguisticDataset
        self.dialogue_samples = self._convert_dataset(raw_dataset)
        print(f"Đã tải và xử lý {len(self.dialogue_samples)} mẫu từ MTOP Intent VN.")

    def _build_vocab_and_intent_map(self, dataset):
        """Xây dựng từ vựng từ dữ liệu và ánh xạ label_text -> target_id."""
        all_tokens = []
        intent_labels = set()
        
        for example in dataset:
            tokens = example["text"].lower().strip().split()
            all_tokens.extend(tokens)
            intent_labels.add(example["label_text"])
        
        vocab_counter = Counter(all_tokens)
        self.vocab = {word: idx for idx, (word, _) in enumerate(sorted(vocab_counter.items()))}
        
        # Ánh xạ label_text sang chỉ số nguyên (dùng chính label có sẵn)
        # Tuy nhiên, để đảm bảo tính nhất quán, ta xây dựng map từ label_text -> label (int)
        self.intent_to_id = {}
        for example in dataset:
            if example["label_text"] not in self.intent_to_id:
                self.intent_to_id[example["label_text"]] = example["label"]
        
        print(f"Tìm thấy {len(self.vocab)} từ vựng và {len(self.intent_to_id)} nhãn intent.")

    def _convert_dataset(self, dataset):
        """Chuyển đổi dataset Hugging Face sang list các dict."""
        converted = []
        for example in dataset:
            # Tạo vector BoW và chiếu xuống bow_dim
            vec_bow = np.zeros(len(self.vocab), dtype=np.float32)
            for token in example["text"].lower().strip().split():
                if token in self.vocab:
                    vec_bow[self.vocab[token]] += 1.0
            
            norm = np.linalg.norm(vec_bow)
            if norm > 0:
                vec_bow = vec_bow / norm
            
            vec_projected = vec_bow @ self.projection_matrix
            
            converted.append({
                "text": example["text"],
                "vector": vec_projected.tolist(),
                "target": self.intent_to_id[example["label_text"]],
                "sub_region": example["label_text"],
            })
        return converted

    def encode_text(self, text: str) -> list:
        """Chuyển đổi một chuỗi văn bản mới thành vector bow_dim chiều."""
        vec_bow = np.zeros(len(self.vocab), dtype=np.float32)
        for token in text.lower().strip().split():
            if token in self.vocab:
                vec_bow[self.vocab[token]] += 1.0
        norm = np.linalg.norm(vec_bow)
        if norm > 0:
            vec_bow = vec_bow / norm
        vec_projected = vec_bow @ self.projection_matrix
        return vec_projected.tolist()

    def get_training_batch(self):
        """Trả về toàn bộ tập dữ liệu mẫu để huấn luyện mạng SNN."""
        return self.dialogue_samples


if __name__ == "__main__":
    dataset = LinguisticDataset(split="train")
    print("Khởi tạo LinguisticDataset thành công!")
    print("Số lượng mẫu hội thoại:", len(dataset.dialogue_samples))
    
    sample = dataset.dialogue_samples[0]
    print("\nMẫu đầu tiên:")
    print("  Text:", sample["text"])
    print("  Vector (8 chiều):", sample["vector"])
    print("  Target ID:", sample["target"])
    print("  Sub-region (Intent):", sample["sub_region"])
    
    new_text = "tôi muốn đặt vé máy bay"
    new_vec = dataset.encode_text(new_text)
    print(f"\nVector cho câu '{new_text}':", new_vec)