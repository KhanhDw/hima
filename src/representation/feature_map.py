from src.representation.latent_vector import LatentVector

class FeatureMap:
    def __init__(self):
        """
        Bản đồ đặc trưng (Feature Map) lưu trữ và quản lý không gian latent vectors 
        để truy hồi và so khớp độ tương đồng giữa các khái niệm/trạng thái.
        """
        self.memory_space = {} # Lưu dạng { "label_name": [latent_vector] }

    def register_feature(self, key, latent_vector):
        """
        Đăng ký hoặc cập nhật một vector đặc trưng cho một khóa (nhãn/khái niệm).
        """
        self.memory_space[key] = latent_vector

    def find_closest(self, input_vector):
        """
        Tìm khóa (key) có latent vector gần nhất với input_vector dựa trên Cosine Similarity.
        Giúp hệ thống nhận diện vật thể/trạng thái tương tự.
        """
        if not self.memory_space:
            return None, 0.0

        best_match = None
        highest_similarity = -1.0

        for key, stored_vector in self.memory_space.items():
            similarity = LatentVector.cosine_similarity(input_vector, stored_vector)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = key

        return best_match, highest_similarity