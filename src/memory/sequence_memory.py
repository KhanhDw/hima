class SequenceMemory:
    def __init__(self, max_history_length=10):
        """
        Bộ nhớ chuỗi (Sequence Memory) lưu trữ lịch sử các vector trạng thái 
        hoặc token theo thứ tự thời gian để phục vụ cho việc dự đoán và phân tích chuỗi.
        """
        self.max_history_length = max_history_length
        self.history = []

    def append(self, latent_vector):
        """
        Thêm một vector mới vào lịch sử chuỗi. Giữ kích thước trong giới hạn max_history_length.
        """
        self.history.append(list(latent_vector))
        if len(self.history) > self.max_history_length:
            self.history.pop(0) # Giữ lại cửa sổ trượt lịch sử gần nhất

    def get_sequence(self):
        """
        Trả về toàn bộ chuỗi lịch sử hiện tại.
        """
        return self.history

    def reset(self):
        """
        Xóa lịch sử chuỗi khi kết thúc chuỗi hiện tại.
        """
        self.history = []