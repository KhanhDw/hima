import random


class Environment:

    def __init__(self):
        self.target_sequence = [0, 1, 2, 1, 0]
        self.current_step_idx = 0
        # Vector trạng thái 8 chiều: 3 chiều đầu cho giá trị target (0,1,2),
        # 5 chiều sau cho vị trí bước (0 đến 4)
        self.current_state_vector = [1, 0, 0, 1, 0, 0, 0, 0]

    def reset(self):
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

        # 2. Mã hóa vị trí bước (5 chiều sau - One-hot encoding cho 5 bước)
        pos_vec = [0, 0, 0, 0, 0]
        if 0 <= self.current_step_idx < 5:
            pos_vec[self.current_step_idx] = 1

        self.current_state_vector = val_vec + pos_vec

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