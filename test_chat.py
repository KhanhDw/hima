from src.integration.main_cognitive_core import MainCognitiveCore

class LinguisticBridge:
    def __init__(self):
        self.core = MainCognitiveCore()
        
        # Từ điển ánh xạ từ vựng đơn giản sang véc-tơ cảm giác 8 chiều
        self.vocab_map = {
            "xin chao": [1.0, 0.2, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0],
            "thoi tiet": [0.0, 1.5, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
            "nang luong": [0.0, 0.0, 0.0, 0.0, 1.8, 1.5, 0.0, 0.0],
            "mac dinh": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        }

    def text_to_vector(self, text):
        text = text.lower().strip()
        for key, vec in self.vocab_map.items():
            if key in text:
                return vec
        return self.vocab_map["mac dinh"] # Mặc định nếu không nhận diện được từ

    def interpret_response(self, cognitive_result):
        # Dựa vào trạng thái tư duy và chunk để dịch ngược ra phản hồi thô
        state = cognitive_result.get("state", "IDLE")
        chunks = cognitive_result.get("focus", {})
        
        if state == "RECHARGING_RESTING":
            return "Hệ thống đang mệt và cần nạp năng lượng (Recharging)..."
        elif state == "EXPLORING_AUTONOMOUSLY":
            return "Hệ thống cảm thấy tò mò và muốn khám phá từ khóa mới!"
        else:
            return f"Hệ thống đã tiếp nhận và xử lý nhịp nhàng (Trạng thái: {state}, Workspace Focus hoạt động)."

    def chat_loop(self):
        print("\n--- BẮT ĐẦU PHÊN GIAO TIẾP VỚI SNN (Gõ 'exit' để thoát) ---")
        while True:
            user_input = input("\nBạn: ")
            if user_input.lower() == 'exit':
                break
            
            # 1. Chuyển text thành véc-tơ cảm giác cho SNN
            sensory_vec = self.text_to_vector(user_input)
            
            # 2. Đưa qua Trái tim nhận thức Cấp 13
            cognitive_result = self.core.step(sensory_vec)
            
            # 3. Đọc kết quả từ mạng và phản hồi
            reply = self.interpret_response(cognitive_result)
            print(f"SNN Core: {reply}")
            print(f"   [Debug Info] State: {cognitive_result.get('state')} | Chunks tích lũy: {len(self.core.chunker.get_chunks())}")

if __name__ == "__main__":
    bridge = LinguisticBridge()
    bridge.chat_loop()