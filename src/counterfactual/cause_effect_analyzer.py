class CauseEffectAnalyzer:
    def __init__(self):
        """
        Bộ phân tích nguyên nhân - kết quả (Cause-Effect Analyzer) so sánh 
        giữa kết quả thực tế và kết quả giả định để tìm ra mối quan hệ nhân quả.
        """
        pass

    def analyze(self, actual_outcome, counterfactual_outcome, alternative_action):
        """
        Phân tích sự khác biệt để đánh giá tác động của hành động thay thế.
        """
        difference_score = 0.0
        if isinstance(actual_outcome, list) and isinstance(counterfactual_outcome, list):
            difference_score = sum(abs(a - c) for a, c in zip(actual_outcome, counterfactual_outcome))
        
        insight = {
            "alternative_action": alternative_action,
            "impact_difference": difference_score,
            "is_better": difference_score > 0.0  # Đánh giá sơ bộ tác động của kịch bản thay thế
        }
        return insight