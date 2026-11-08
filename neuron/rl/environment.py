import random

class Environment:

    def __init__(self):
        # Định nghĩa một chuỗi mục tiêu tuần tự (Goal-Directed Sequence)
        # Ví dụ: Agent cần lần lượt hoàn thành chuỗi state 0 -> 1 -> 2
        self.target_sequence = [0, 1, 2]
        self.current_step_idx = 0
        self.current_state_vector = [1, 0, 0] # Bắt đầu ở trạng thái tương ứng với 0

    def reset(self):
        """Reset lại môi trường về bước đầu tiên của chuỗi"""
        self.current_step_idx = 0
        self.current_state_vector = [1, 0, 0]
        return self.current_state_vector, self.target_sequence[self.current_step_idx]

    def observe(self):
        """
        Trả về vector trạng thái hiện tại và mục tiêu cần đạt ở bước này 
        để tương thích ngược với các hàm gọi cũ.
        """
        target = self.target_sequence[self.current_step_idx]
        return self.current_state_vector, target

    def step(self, action):
        """
        Thực hiện một action, kiểm tra đúng/sai trong chuỗi tuần tự theo thời gian.
        Trả về: (next_state_vector, reward, done)
        """
        expected_target = self.target_sequence[self.current_step_idx]
        
        if action == expected_target:
            reward = 1.0
            self.current_step_idx += 1
            
            # Kiểm tra xem đã hoàn thành toàn bộ chuỗi chưa
            if self.current_step_idx >= len(self.target_sequence):
                done = True
                self.current_step_idx = 0
            else:
                done = False
                
            # Cập nhật vector trạng thái tiếp theo tương ứng với target mới trong chuỗi
            if self.current_step_idx == 0:
                self.current_state_vector = [1, 0, 0]
            elif self.current_step_idx == 1:
                self.current_state_vector = [0, 1, 0]
            elif self.current_step_idx == 2:
                self.current_state_vector = [0, 0, 1]
                
        else:
            reward = -1.0
            done = True # Sai ở bước nào, kết thúc chuỗi và reset về đầu
            self.current_step_idx = 0
            self.current_state_vector = [1, 0, 0]

        return self.current_state_vector, reward, done

    def reward(self, action, target):
        """Giữ lại để tương thích với các module đánh giá cũ"""
        if action == target:
            return 1.0
        return -1.0

    def evaluate(self, action, target):
        """Giữ lại để tương thích với các module đánh giá cũ"""
        reward = self.reward(action, target)
        done = reward > 0
        return reward, done