from collections import deque


class TransitionMemory:

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.transitions = deque(maxlen=max_size)

    def store(
        self,
        state,
        action,
        next_state,
        reward=0.0
    ):
        self.transitions.append({
            "state": list(state),
            "action": action,
            "next_state": list(next_state),
            "reward": reward
        })

    def get_all(self):
        return list(self.transitions)

    def find_matches(
        self,
        state,
        action
    ):
        matches = []

        for t in self.transitions:
            if t["state"] == state and t["action"] == action:
                matches.append(t)

        return matches

    def size(self):
        return len(self.transitions)