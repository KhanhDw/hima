class PerformanceMonitor:
    def __init__(self):
        """
        Bộ giám sát hiệu suất (Performance Monitor) theo dõi liên tục các chỉ số 
        về độ chính xác, sai số và phần thưởng của hệ thống theo thời gian thực.
        """
        self.metrics_history = []

    def record_metrics(self, accuracy, error, reward):
        """
        Ghi nhận bộ chỉ số của một chu kỳ hoạt động hoặc episode mới.
        """
        metrics = {
            "accuracy": accuracy,
            "error": error,
            "reward": reward
        }
        self.metrics_history.append(metrics)
        
        # Giữ lại lịch sử gần nhất để phân tích xu hướng
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)

    def get_summary(self):
        """
        Tổng hợp các chỉ số trung bình gần đây để phục vụ cho việc tự tối ưu hóa.
        """
        if not self.metrics_history:
            return {"avg_accuracy": 0.0, "avg_error": 0.0, "avg_reward": 0.0}
            
        n = len(self.metrics_history)
        avg_acc = sum(m["accuracy"] for m in self.metrics_history) / n
        avg_err = sum(m["error"] for m in self.metrics_history) / n
        avg_rew = sum(m["reward"] for m in self.metrics_history) / n
        
        return {
            "avg_accuracy": avg_acc,
            "avg_error": avg_err,
            "avg_reward": avg_rew
        }