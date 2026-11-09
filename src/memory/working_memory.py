class WorkingMemory:
    def __init__(self, capacity=5):
        self.capacity = capacity
        self.slots = []   # [(importance, item), ...]

    def load(self, item, importance=1.0):
        self.slots.append((importance, item))
        if len(self.slots) > self.capacity:
            self.slots.sort(key=lambda x: -x[0])
            self.slots = self.slots[:self.capacity]

    def get_contents(self):
        return [item for _, item in self.slots]

    def get_prioritized(self):
        return [(imp, item) for imp, item in self.slots]

    def __len__(self):
        return len(self.slots)

    def clear(self):
        self.slots = []