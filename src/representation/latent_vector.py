import math

class LatentVector:
    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """
        Tính độ tương đồng góc giữa hai latent vector (giúp hệ thống nhận biết 
        'vật này giống vật kia' dựa trên góc không gian ẩn).
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
        magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))
        
        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0
            
        return dot_product / (magnitude_a * magnitude_b)

    @staticmethod
    def euclidean_distance(vec_a, vec_b):
        """
        Tính khoảng cách Euclidean giữa hai latent vector trong không gian đặc trưng.
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return float('inf')
            
        sum_sq_diff = sum((a - b) ** 2 for a, b in zip(vec_a, vec_b))
        return math.sqrt(sum_sq_diff)