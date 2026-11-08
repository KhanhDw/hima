class Clock:
    def __init__(self):
        self.time = 0

    def tick(self):
        self.time += 1
        return self.time

    def reset(self):
        self.time = 0