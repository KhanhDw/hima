class CounterfactualEngine:
    def __init__(self):
        """
        Bộ máy suy luận phản thực tế (Counterfactual Engine) mô phỏng các kịch bản 
        'giá như...' (What if) bằng cách thay đổi hành động hoặc trạng thái trong quá khứ.
        """
        pass

    def simulate_alternative(self, past_state, alternative_action, environment_model=None):
        """
        Mô phỏng kết quả giả định nếu trong quá khứ thực hiện một hành động khác.
        """
        if environment_model and hasattr(environment_model, "predict_transition"):
            return environment_model.predict_transition(past_state, alternative_action)
            
        # Mô phỏng phản ứng giả định cơ bản dựa trên trạng thái cũ và hành động thay thế
        simulated_next_state = [min(1.0, max(0.0, s + (0.1 if alternative_action is not None else 0.0))) for s in past_state] if isinstance(past_state, list) else past_state
        return simulated_next_state