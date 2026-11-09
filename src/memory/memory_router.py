class MemoryRouter:
    def __init__(self, sequence, working, context,
                 temporal, episodic=None, consolidator=None):
        self.sequence = sequence
        self.working = working
        self.context = context
        self.temporal = temporal
        self.episodic = episodic
        self.consolidator = consolidator

    def observe(self, state, action, context, reward,
                next_state, done=False):
        """Một điểm vào duy nhất cho mỗi transition."""
        self.sequence.append(state)
        self.working.load(state, importance=abs(reward) + 0.5)
        self.temporal.update(state)
        for k, v in (context or {}).items():
            self.context.set_context(k, v)
        if self.episodic is not None:
            self.episodic.store(state, action, context,
                                reward, next_state, done)

    def recall_similar(self, state, top_k=5):
        if self.episodic is None:
            return []
        return self.episodic.query_similar(state, top_k=top_k)

    def end_episode(self):
        self.context.commit()
        self.sequence.reset()
        self.temporal.reset()
        if self.consolidator is not None:
            self.consolidator.maybe_consolidate()