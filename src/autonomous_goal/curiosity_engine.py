class CuriosityEngine:
    def __init__(self):
        """
        Bộ máy tò mò (Curiosity Engine) đánh giá độ mới lạ (novelty) 
        và mức độ bất định của trạng thái hiện tại để kích thích khám phá.
        """
        self.visited_states = set()

    def calculate_curiosity(self, state_vector):
        """
        Tính điểm tò mò dựa trên mức độ quen thuộc của trạng thái.
        Trạng thái càng ít gặp, điểm tò mò càng cao.
        """
        if not state_vector:
            return 1.0

        # Chuyển state_vector thành tuple để dễ dàng lưu trữ và kiểm tra trong set
        state_key = tuple(round(val, 1) for val in state_vector) if isinstance(state_vector, list) else state_vector
        
        if state_key not in self.visited_states:
            self.visited_states.add(state_key)
            return 1.0  # Trạng thái hoàn toàn mới -> Tò mò tối đa
            
        return 0.1  # Trạng thái quen thuộc -> Điểm tò mò thấp