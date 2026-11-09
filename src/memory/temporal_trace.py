class TemporalTrace:
    def __init__(self, decay_rate=0.8, mode="ewma"):
        """
        mode:
          - "ewma": trace = d·old + (1-d)·new  → bounded, hội tụ về input
          - "eligibility": trace = d·old + new → TD-λ, hội tụ về input/(1-d)
        """
        self.decay_rate = decay_rate
        self.mode = mode
        self.current_trace = []

    def update(self, current_vector):
        if not self.current_trace or len(self.current_trace) != len(current_vector):
            self.current_trace = list(current_vector)
            return self.current_trace

        d = self.decay_rate
        alpha = (1 - d) if self.mode == "ewma" else 1.0

        self.current_trace = [
            round(d * old + alpha * new, 4)
            for old, new in zip(self.current_trace, current_vector)
        ]
        return self.current_trace