import torch
import numpy as np

class LinguisticDataset:
    def __init__(self):
        """
        Bộ dữ liệu ngôn ngữ cấu trúc dành riêng cho hệ thống Spiking Neural Network (SNN).
        Ánh xạ ngữ nghĩa ngôn ngữ thành các vector cảm giác 8 chiều và chuỗi mục tiêu.
        """
        # 1. Bảng ánh xạ Từ vựng & Ngữ nghĩa (8-dimensional semantic vectors)
        # 8 chiều đại diện cho: [Chào hỏi, Hỏi thăm, Yêu cầu, Cảm xúc, Năng lượng, Cần giúp đỡ, Tạm biệt, Mặc định]
        self.vocab = {
            "xin chao":    [1.0, 0.0, 0.0, 0.5, 0.2, 0.0, 0.0, 0.0],
            "chao bạn":    [0.9, 0.1, 0.0, 0.4, 0.3, 0.0, 0.0, 0.0],
            "ban khoe khong": [0.2, 1.0, 0.0, 0.3, 0.5, 0.0, 0.0, 0.0],
            "thoi tiet":   [0.0, 0.2, 0.8, 0.0, 0.1, 0.0, 0.0, 0.0],
            "nang luong":  [0.0, 0.0, 0.3, 0.0, 1.0, 0.2, 0.0, 0.0],
            "giup toi":    [0.0, 0.0, 1.0, 0.2, 0.1, 1.0, 0.0, 0.0],
            "tam biet":    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0],
            "mac dinh":    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        }

        # 2. Tập dữ liệu Chuỗi Hội thoại (Conversational Sequences)
        # Format: (Input Text, Target Action/Output Neuron Index, Expected State)
        # Target Index: 0 -> Response A (Chào lại), 1 -> Response B (Thông tin/Giúp đỡ), 2 -> Response C (Nghỉ ngơi/Tạm biệt)
        self.dialogue_samples = [
            {"input": "xin chao", "target": 0, "sub_region": "Task_0"},
            {"input": "chao bạn", "target": 0, "sub_region": "Task_0"},
            {"input": "ban khoe khong", "target": 0, "sub_region": "Task_0"},
            {"input": "thoi tiet", "target": 1, "sub_region": "Task_1"},
            {"input": "giup toi", "target": 1, "sub_region": "Task_1"},
            {"input": "nang luong", "target": 2, "sub_region": "Task_2"},
            {"input": "tam biet", "target": 2, "sub_region": "Task_2"}
        ]

    def encode_text(self, text: str) -> list:
        """
        Chuyển đổi chuỗi văn bản nhập vào thành vector 8 chiều.
        """
        text_clean = text.lower().strip()
        for key, vec in self.vocab.items():
            if key in text_clean:
                return vec
        return self.vocab["mac dinh"]

    def get_training_batch(self):
        """
        Trả về toàn bộ tập dữ liệu mẫu để huấn luyện mạng SNN.
        """
        batch = []
        for sample in self.dialogue_samples:
            vec = self.encode_text(sample["input"])
            batch.append({
                "text": sample["input"],
                "vector": vec,
                "target": sample["target"],
                "sub_region": sample["sub_region"]
            })
        return batch


if __name__ == "__main__":
    dataset = LinguisticDataset()
    print("Khởi tạo LinguisticDataset thành công!")
    print("Số lượng mẫu hội thoại:", len(dataset.dialogue_samples))
    sample_vec = dataset.encode_text("xin chao")
    print("Vector mã hóa cho 'xin chao':", sample_vec)