class GoalManager:
    def __init__(self):
        """
        Quản lý mục tiêu (Goal Manager) thiết lập, theo dõi và cập nhật 
        các trạng thái đích mà hệ thống cần hướng tới trong quá trình vận hành.
        """
        self.current_goal = None
        self.sub_goals = []
        self.is_achieved = False

    def set_goal(self, goal_vector):
        """
        Thiết lập mục tiêu mới cho hệ thống.
        """
        self.current_goal = list(goal_vector)
        self.is_achieved = False

    def check_achieved(self, current_state_vector, tolerance=0.1):
        """
        Kiểm tra xem trạng thái hiện tại đã đạt được mục tiêu hay chưa 
        dựa trên sai số cho phép.
        """
        if not self.current_goal or not current_state_vector:
            return False
            
        if len(self.current_goal) != len(current_state_vector):
            return False

        # Tính khoảng cách đơn giản giữa trạng thái hiện tại và mục tiêu
        diff = sum(abs(g - c) for g, c in zip(self.current_goal, current_state_vector))
        if diff <= tolerance:
            self.is_achieved = True
            
        return self.is_achieved

    def get_goal(self):
        """
        Truy xuất mục tiêu hiện tại.
        """
        return self.current_goal