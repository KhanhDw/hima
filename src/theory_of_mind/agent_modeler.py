class AgentModeler:
    def __init__(self):
        """
        Mô hình hóa tác nhân (Agent Modeler) lưu trữ và cập nhật hồ sơ hành vi 
        của các tác nhân khác trong môi trường (Theory of Mind).
        """
        self.agent_profiles = {}

    def update_profile(self, agent_id, observed_action, observed_state):
        """
        Ghi nhận hành động và trạng thái quan sát được từ một agent khác 
        để cập nhật mô hình hành vi của họ.
        """
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = {
                "action_history": [],
                "state_history": [],
                "inferred_goal": None
            }
            
        profile = self.agent_profiles[agent_id]
        profile["action_history"].append(observed_action)
        profile["state_history"].append(observed_state)
        
        # Giữ giới hạn lịch sử gần nhất
        if len(profile["action_history"]) > 20:
            profile["action_history"].pop(0)
            profile["state_history"].pop(0)

    def get_profile(self, agent_id):
        """
        Truy xuất hồ sơ mô hình tâm lý của một agent cụ thể.
        """
        return self.agent_profiles.get(agent_id, None)