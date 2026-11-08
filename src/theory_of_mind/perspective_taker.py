class PerspectiveTaker:
    def __init__(self):
        """
        Bộ hoán đổi góc nhìn (Perspective Taker) cho phép hệ thống mô phỏng 
        thế giới quan và trạng thái dưới lăng kính của một tác nhân khác.
        """
        pass

    def adopt_perspective(self, agent_profile, current_environment_state):
        """
        Mô phỏng trạng thái môi trường từ góc nhìn giả định của agent khác.
        Ở mức cơ bản, trả về bản sao trạng thái kèm theo thông tin mô hình của agent.
        """
        if not agent_profile:
            return current_environment_state
            
        # Mô phỏng góc nhìn bằng cách kết hợp trạng thái môi trường và thông tin agent
        simulated_perspective = {
            "environment_state": current_environment_state,
            "agent_known_state": agent_profile.get("state_history", [])[-1:] if agent_profile.get("state_history") else None
        }
        return simulated_perspective