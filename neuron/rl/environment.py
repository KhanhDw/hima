import random


class Environment:

    def __init__(self):
        self.branch_a = [0, 1, 2, 1, 0]
        self.branch_b = [0, 1, 0, 2, 0]
        
        self.current_branch = self.branch_a
        self.target_sequence = self.current_branch
        self.current_step_idx = 0
        self.branch_id = 0 # 0 cho Branch A, 1 cho Branch B
        
        # Mở rộng vector trạng thái lên 9 chiều: 
        # 3 chiều đầu (giá trị), 5 chiều sau (vị trí bước), 1 chiều cuối (Branch Flag)
        self.current_state_vector = [1, 0, 0, 1, 0, 0, 0, 0, 1]

    def reset(self):
        self.branch_id = random.choice([0, 1])
        self.current_branch = self.branch_a if self.branch_id == 0 else self.branch_b
        self.target_sequence = self.current_branch
        self.current_step_idx = 0
        self.update_state_vector()
        return self.current_state_vector, self.target_sequence[self.current_step_idx]

    def update_state_vector(self):
        target = self.target_sequence[self.current_step_idx]

        # 1. Mã hóa giá trị target (3 chiều đầu)
        val_vec = [0, 0, 0]
        if target == 0:
            val_vec = [1, 0, 0]
        elif target == 1:
            val_vec = [0, 1, 0]
        elif target == 2:
            val_vec = [0, 0, 1]

        # 2. Mã hóa vị trí bước (5 chiều tiếp theo - One-hot encoding)
        pos_vec = [0, 0, 0, 0, 0]
        if 0 <= self.current_step_idx < len(self.target_sequence):
            pos_vec[self.current_step_idx] = 1

        # 3. Mã hóa Branch Flag (1 chiều cuối: [1, 0] cho nhánh A, [0, 1] hoặc giá trị phân biệt cho nhánh B)
        # Ở đây ta dùng 1 chiều đơn giản: 1 nếu là nhánh A, 0 nếu là nhánh B
        branch_vec = [1 if self.branch_id == 0 else 0]

        self.current_state_vector = val_vec + pos_vec + branch_vec

    def observe(self):
        target = self.target_sequence[self.current_step_idx]
        return self.current_state_vector, target

    def step(self, action):
        expected_target = self.target_sequence[self.current_step_idx]

        if action == expected_target:
            reward = 1.0
            self.current_step_idx += 1

            if self.current_step_idx >= len(self.target_sequence):
                done = True
                self.current_step_idx = 0
            else:
                done = False

            self.update_state_vector()
        else:
            reward = -1.0
            done = True
            self.current_step_idx = 0
            self.update_state_vector()

        return self.current_state_vector, reward, done

    def reward(self, action, target):
        if action == target:
            return 1.0
        return -1.0

    def evaluate(self, action, target):
        reward = self.reward(action, target)
        done = reward > 0
        return reward, done