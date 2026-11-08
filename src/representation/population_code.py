import math

class PopulationCoder:
    def __init__(self, num_neurons=10, min_val=0.0, max_val=2.0, sigma=0.5):
        """
        Mã hóa quần thể (Population Coding) chuyển đổi một giá trị scalar thô
        thành một phân phối kích thích trên một quần thể nơ-ron.
        """
        self.num_neurons = num_neurons
        self.min_val = min_val
        self.max_val = max_val
        self.sigma = sigma
        
        # Chia đều các tâm tiếp nhận (preferred values) cho các nơ-ron trong quần thể
        self.centers = [
            min_val + i * (max_val - min_val) / (num_neurons - 1)
            if num_neurons > 1 else min_val
            for i in range(num_neurons)
        ]

    def encode(self, value):
        """
        Nhận vào một giá trị (ví dụ: nhãn 0, 1, 2 hoặc một giá trị liên tục)
        và trả về vector kích thích quần thể (Latent Vector dạng float từ 0.0 đến 1.0).
        """
        encoded_vector = []
        for center in self.centers:
            # Hàm Gaussian tuning curve tính độ kích hoạt của nơ-ron theo khoảng cách tới tâm
            activation = math.exp(-((value - center) ** 2) / (2 * (self.sigma ** 2)))
            encoded_vector.append(round(activation, 4))
        return encoded_vector