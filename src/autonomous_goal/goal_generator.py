class GoalGenerator:
    def __init__(self):
        """
        Bộ sinh mục tiêu tự động (Goal Generator) tạo ra các mục tiêu mới 
        dựa trên mức độ tò mò và trạng thái khám phá của hệ thống.
        """
        self.active_autonomous_goals = []

    def evaluate_and_generate(self, curiosity_score, current_context):
        """
        Nếu điểm tò mò vượt ngưỡng, tự động sinh ra mục tiêu khám phá vùng đất mới.
        """
        if curiosity_score > 0.5:
            new_goal = {
                "goal_id": f"auto_goal_{len(self.active_autonomous_goals) + 1}",
                "description": "Explore novel state space and reduce uncertainty",
                "priority": curiosity_score,
                "context": current_context
            }
            self.active_autonomous_goals.append(new_goal)
            return new_goal
            
        return None