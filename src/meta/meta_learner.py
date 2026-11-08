class MetaLearner:
    def __init__(self, initial_learning_rate=0.01):
        """
        Bộ siêu học tập (Meta-Learner) điều chỉnh các siêu tham số của hệ thống 
        (như tốc độ học, hệ số điều biến) dựa trên xu hướng hiệu suất tổng thể.
        """
        self.learning_rate = initial_learning_rate
        self.performance_history = []

    def adapt(self, recent_accuracy):
        """
        Tự động điều chỉnh siêu tham số dựa trên độ chính xác gần đây.
        Nếu độ chính xác giảm, thực hiện tinh chỉnh tốc độ học để thích ứng.
        """
        self.performance_history.append(recent_accuracy)
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)

        # Đánh giá xu hướng: nếu hiệu suất thấp, tăng tốc độ học để thích ứng nhanh hơn
        avg_acc = sum(self.performance_history) / len(self.performance_history)
        if avg_acc < 0.5:
            self.learning_rate = min(0.1, self.learning_rate * 1.05)
        else:
            self.learning_rate = max(0.001, self.learning_rate * 0.95)
            
        return self.learning_rate