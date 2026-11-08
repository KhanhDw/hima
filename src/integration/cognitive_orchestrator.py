class CognitiveOrchestrator:
    def __init__(self, global_workspace):
        """
        Nhạc trưởng điều phối nhận thức (Cognitive Orchestrator) quản lý vòng lặp
        vận hành toàn cục, điều hướng luồng thông tin giữa các phân vùng nhận thức.
        """
        self.workspace = global_workspace
        self.active_state = "IDLE"

    def tick(self, sensory_input, drives_status, curiosity_score):
        """
        Thực hiện một nhịp nhận thức toàn cục (cognitive tick):
        1. Đánh giá động lực và tò mò.
        2. Cập nhật không gian làm việc toàn cục.
        3. Định hướng hành vi phản ứng của toàn hệ thống.
        """
        # Đưa thông tin cảm giác vào không gian ý thức chung
        conscious_packet = self.workspace.broadcast(sensory_input, source_module="SensoryCore")
        
        # Quyết định trạng thái hành vi dựa trên nội lực và tò mò
        if drives_status.get("energy", 1.0) < 0.3:
            self.active_state = "RECHARGING_RESTING"
        elif curiosity_score > 0.5:
            self.active_state = "EXPLORING_AUTONOMOUSLY"
        else:
            self.active_state = "GOAL_DIRECTED_EXECUTION"
            
        return {
            "state": self.active_state,
            "focus": conscious_packet
        }