from collections import Counter


class StatePredictor:

    def __init__(self, transition_memory):
        self.memory = transition_memory

    def predict(
        self,
        state,
        action
    ):
        matches = self.memory.find_matches(
            state,
            action
        )

        if not matches:
            return None, 1.0

        next_states = [
            tuple(t["next_state"])
            for t in matches
        ]

        prediction = Counter(
            next_states
        ).most_common(1)[0][0]

        confidence = len(matches) / (
            len(matches) + 1
        )

        return (
            list(prediction),
            round(confidence, 4)
        )