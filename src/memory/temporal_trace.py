class TemporalTrace:
    def __init__(self, decay_rate=0.8):
        """
        Quản lý dấu vết thời gian (Temporal Trace / Eligibility Trace) 
        để lưu giữ dư âm kích thích qua các bước thời gian.
        """
        self.decay_rate = decay_rate
        self.current_trace = []

    def update(self, current_vector):
        """
        Cập nhật dấu vết thời gian: mờ dần (decay) dấu vết cũ và cộng thêm tín hiệu mới.
        Trace = Trace_cũ * decay_rate + Vector_mới
        """
        if not self.current_trace:
            self.current_trace = list(current_vector)
            return self.current_trace

        # Đảm bảo kích thước khớp nhau
        if len(self.current_trace) != len(current_vector):
            self.current_trace = list(current_vector)
            return self.current_trace

        updated = []
        for old_val, new_val in zip(self.current_trace, current_vector):
            decayed_val = (old_val * self.decay_rate) + new_val
            updated.append(round(decayed_val, 4))
            
        self.current_trace = updated
        return self.current_trace

    def reset(self):
        """
        Làm sạch dấu vết khi bắt đầu một episode hoặc chuỗi mới.
        """
        self.current_trace = []