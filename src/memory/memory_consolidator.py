class MemoryConsolidator:
    """
    Level 14.6: chạy mỗi N episode, nén episodic → pattern.
    """

    def __init__(self, episodic_memory, world_model, trigger_every=50):
        self.episodic = episodic_memory
        self.world_model = world_model
        self.trigger_every = trigger_every
        self._counter = 0
        self.patterns = []   # [(signature, count, avg_reward)]

    def maybe_consolidate(self):
        self._counter += 1
        if self._counter < self.trigger_every:
            return
        self._counter = 0
        self._consolidate()

    def _consolidate(self):
        if len(self.episodic.buffer) < 100:
            return

        # Gom theo (quantized_state, action)
        buckets = {}
        for ep in self.episodic.buffer:
            sig = (
                tuple(round(v, 1) for v in ep["state"]),
                ep["action"],
            )
            buckets.setdefault(sig, []).append(ep["reward"])

        self.patterns = [
            (sig, len(rs), sum(rs) / len(rs))
            for sig, rs in buckets.items()
            if len(rs) >= 3
        ]
        self.patterns.sort(key=lambda x: -x[1])
        self.patterns = self.patterns[:500]   # giữ top