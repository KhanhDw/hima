class Predictor:
    def __init__(self):
        """
        Bộ dự đoán (Predictor) phân tích chuỗi lịch sử và ngữ cảnh 
        để dự đoán trạng thái hoặc token tiếp theo của dòng thời gian.
        """
        self.last_prediction = None

    def predict(self, sequence_history, context=None):
        """
        Dựa vào chuỗi lịch sử đầu vào, ước tính trạng thái tiếp theo.
        Ở mức cơ bản, nếu chuỗi có lịch sử, ta có thể ngoại suy hoặc dự đoán 
        dựa trên xu hướng gần nhất.
        """
        if not sequence_history:
            return None

        # Logic dự đoán cơ bản: Kế thừa hoặc tinh chỉnh từ phần tử gần nhất trong chuỗi
        # (Sẽ được mở rộng thành mô hình trọng số SNN/Linear ở các bước tinh chỉnh sâu hơn)
        latest_item = sequence_history[-1]
        
        # Tạo bản sao dự đoán cho bước tiếp theo
        self.last_prediction = list(latest_item)
        return self.last_prediction

    def get_last_prediction(self):
        """
        Lấy kết quả dự đoán gần nhất.
        """
        return self.last_prediction