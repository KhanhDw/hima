class DriveMotivator:
    def __init__(self):
        """
        Bộ quản lý động lực nội tại (Drive Motivator) duy trì các trạng thái động lực 
        cơ bản như mức năng lượng, cân bằng nội môi và khát vọng khám phá.
        """
        self.drives = {
            "energy": 1.0,
            "curiosity_drive": 0.8,
            "competence": 0.5
        }

    def update_drives(self, activity_cost=0.05, reward_signal=0.0):
        """
        Cập nhật các giá trị động lực nội tại dựa trên hoạt động hiện tại và phần thưởng.
        """
        # Tiêu hao năng lượng theo hoạt động
        self.drives["energy"] = max(0.1, self.drives["energy"] - activity_cost)
        
        # Củng cố năng lực dựa trên phần thưởng nhận được
        if reward_signal > 0:
            self.drives["competence"] = min(1.0, self.drives["competence"] + 0.1)
            
        return self.drives

    def need_rest_or_recharge(self):
        """
        Kiểm tra xem hệ thống có đang cạn kiệt năng lượng hay không.
        """
        return self.drives["energy"] < 0.3