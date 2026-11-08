class SurpriseCalculator:
    @staticmethod
    def calculate(prediction_error):
        """
        Đo lường mức độ bất ngờ (Surprise) dựa trên sai số dự đoán.
        Sai số càng lớn nghĩa là thực tế đi lệch hoàn toàn so với kỳ vọng,
        tạo ra mức độ bất ngờ cao, kích thích hệ thống cập nhật trọng số và ghi nhớ.
        """
        if prediction_error is None:
            return 0.0
        
        # Mức độ bất ngờ tỷ lệ thuận với sai số dự đoán
        surprise_value = max(0.0, float(prediction_error))
        return round(surprise_value, 4)