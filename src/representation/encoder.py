from src.representation.population_code import PopulationCoder

class Encoder:
    def __init__(self, num_neurons=10, min_val=0.0, max_val=2.0, sigma=0.5):
        """
        Bộ mã hóa (Encoder) tổng hợp các chiến lược biểu diễn, 
        chuyển đổi dữ liệu thô từ môi trường thành các Latent Vector phân tán.
        """
        self.population_coder = PopulationCoder(
            num_neurons=num_neurons, 
            min_val=min_val, 
            max_val=max_val, 
            sigma=sigma
        )
        self.last_latent_vector = []

    def encode_state(self, raw_state_or_value):
        """
        Mã hóa một trạng thái hoặc giá trị thô thành latent vector phân tán.
        Hỗ trợ cả giá trị scalar đơn lẻ lẫn danh sách/vector thô.
        """
        if isinstance(raw_state_or_value, (int, float)):
            self.last_latent_vector = self.population_coder.encode(float(raw_state_or_value))
        elif isinstance(raw_state_or_value, list):
            # Nếu là list (như state vector cũ), ta có thể mã hóa từng phần tử hoặc gộp lại
            # Ở mức cơ bản, nếu là vector one-hot hoặc nhiều chiều, ta trích xuất giá trị hoặc mã hóa gộp
            encoded_parts = []
            for val in raw_state_or_value:
                encoded_parts.extend(self.population_coder.encode(float(val)))
            self.last_latent_vector = encoded_parts
        else:
            raise TypeError("Unsupported input type for Encoder. Expected int, float, or list.")
            
        return self.last_latent_vector

    def get_latent_vector(self):
        """
        Trả về latent vector gần nhất được tạo ra.
        """
        return self.last_latent_vector