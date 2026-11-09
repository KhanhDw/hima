class CausalDiscovery:

    def __init__(self):
        self.causal_pairs = {}

    def observe(
        self,
        state,
        action,
        next_state
    ):
        key = (
            tuple(state),
            action
        )

        self.causal_pairs[key] = \
            self.causal_pairs.get(
                key,
                0
            ) + 1

    def get_causal_strength(
        self,
        state,
        action
    ):
        key = (
            tuple(state),
            action
        )

        return self.causal_pairs.get(
            key,
            0
        )