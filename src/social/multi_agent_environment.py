class MultiAgentEnvironment:
    def __init__(self, agent_ids):
        """
        Môi trường đa tác nhân (Multi-Agent Environment) quản lý không gian chung,
        trạng thái của các tác nhân và điều phối vòng lặp tương tác xã hội.
        """
        self.agents = agent_ids
        self.global_state = {agent_id: None for agent_id in agent_ids}  # Sửa 'agent_ids' thành 'agent_id'

    def step(self, actions_dict):
        """
        Thực hiện một bước môi trường dựa trên hành động đồng thời của tất cả các tác nhân.
        actions_dict: {agent_id: action}
        """
        rewards = {}
        for agent_id, action in actions_dict.items():
            # Đánh giá phần thưởng cơ bản cho từng tác nhân trong không gian chung
            rewards[agent_id] = 1.0 if action is not None else 0.0
            
        return self.global_state, rewards