class ConfidenceEstimator:
    @staticmethod
    def estimate(prediction_error, max_error_bound=2.0):
        """
        Đánh giá độ tin cậy (Confidence) của dự đoán dựa trên sai số dự đoán.
        Sai số càng nhỏ thì độ tin cậy càng cao (tiệm cận 1.0).
        """
        if prediction_error is None:
            return 0.0
        
        # Chuẩn hóa độ tin cậy: Sai số càng lớn, confidence càng giảm về 0
        confidence = max(0.0, 1.0 - (prediction_error / max_error_bound))
        return round(confidence, 4)