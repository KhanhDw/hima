class Planner:
    def __init__(self):
        """
        Bộ lập kế hoạch (Planner) xây dựng lộ trình dự phóng chuỗi hành động 
        từ trạng thái hiện tại hướng tới mục tiêu (Goal).
        """
        self.planned_path = []

    def generate_plan(self, current_state, goal_state, sequence_history=None):
        """
        Lập kế hoạch chuyển dịch từ trạng thái hiện tại sang mục tiêu.
        Ở mức cơ bản, tạo chuỗi các bước định hướng hành động.
        """
        if not current_state or not goal_state:
            return []
            
        # Xây dựng lộ trình nội suy cơ bản
        self.planned_path = [current_state, goal_state]
        return self.planned_path

    def get_next_action_step(self):
        """
        Lấy bước hành động tiếp theo trong kế hoạch.
        """
        if self.planned_path:
            return self.planned_path.pop(0)
        return None