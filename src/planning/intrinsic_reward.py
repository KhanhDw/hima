class IntrinsicRewardCalculator:
    @staticmethod
    def calculate(previous_distance, current_distance):
        """
        Tính toán phần thưởng nội tại (Intrinsic Reward) dựa trên tiến độ tiến gần đến mục tiêu.
        Nếu khoảng cách tới mục tiêu giảm đi, hệ thống nhận phần thưởng dương (khuyến khích tiến bộ),
        ngược lại nhận giá trị âm hoặc bằng 0.
        """
        if previous_distance is None or current_distance is None:
            return 0.0
            
        # Khoảng cách giảm nghĩa là đang tiến gần đến mục tiêu -> phần thưởng dương
        progress = previous_distance - current_distance
        return round(float(progress), 4)