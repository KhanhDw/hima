class IntentPredictor:
    def __init__(self):
        """
        Bộ dự đoán ý định (Intent Predictor) phân tích lịch sử hành vi của agent khác
        để dự phóng hành động hoặc đích đến tiếp theo của họ.
        """
        pass

    def predict_intent(self, agent_profile):
        """
        Dựa vào lịch sử hành động của agent để dự đoán hành động tiếp theo.
        Nếu có đủ dữ liệu, trả về hành động xuất hiện thường xuyên nhất gần đây.
        """
        if not agent_profile or not agent_profile.get("action_history"):
            return None
            
        history = agent_profile["action_history"]
        # Dự đoán cơ bản bằng cách lặp lại xu hướng hành động gần nhất
        return history[-1]