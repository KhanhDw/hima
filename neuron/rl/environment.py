import random


class Environment:

    def __init__(self):
        # Định nghĩa các cặp (vector đầu vào, target)
        self.patterns = [
            ([1, 0, 0], 0),
            ([0, 1, 0], 1),
            ([0, 0, 1], 2),
        ]

    def observe(self):
        """
        Trả về ngẫu nhiên một mẫu gồm (vector_input, target).
        """
        return random.choice(self.patterns)

    def reward(self, action, target):
        """
        +1 nếu đúng, -1 nếu sai
        """
        if action == target:
            return 1.0
        return -1.0

    def evaluate(self, action, target):
        reward = self.reward(action, target)
        done = reward > 0
        return reward, done