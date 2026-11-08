class BaseSoma:
    def __init__(self, refractory_period=2):
        self.refractory_period = refractory_period
        self.refractory_left = 0
        self.V = 0.0

    def reset(self):
        self.V = 0.0
        self.refractory_left = 0