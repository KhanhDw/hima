import math

class PredictionError:
    @staticmethod
    def calculate(predicted_vector, actual_vector):
        """
        Tính toán khoảng cách sai lệch (Sai số dự đoán) giữa trạng thái dự đoán 
        và trạng thái thực tế xảy ra trong môi trường.
        """
        if not predicted_vector or not actual_vector or len(predicted_vector) != len(actual_vector):
            return 1.0 # Sai số tối đa nếu thiếu dữ liệu hoặc lệch chiều
            
        # Tính sai số bình phương trung bình (MSE) hoặc Euclidean distance
        sum_sq_diff = sum((p - a) ** 2 for p, a in zip(predicted_vector, actual_vector))
        mse = sum_sq_diff / len(predicted_vector)
        
        return round(math.sqrt(mse), 4) # Trả về giá trị sai số chuẩn hóa