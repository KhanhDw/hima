from .base_soma import BaseSoma

class AdaptiveLIFSoma(BaseSoma):
    def __init__(self, theta=1.0, leak=0.9, a=0.01, b=0.2, tau_w=10.0, refractory_period=2):
        super().__init__(refractory_period)
        self.theta = theta
        self.leak = leak
        self.a = a
        self.b = b
        self.tau_w = tau_w
        self.w = 0.0

    def step(self, current):
        if self.refractory_left > 0:
            self.refractory_left -= 1
            return 0

        dw = (self.a * self.V - self.w) / self.tau_w
        self.w += dw
        self.V = self.V * self.leak + current - self.w

        if self.V >= self.theta:
            self.V = 0.0
            self.w += self.b
            self.refractory_left = self.refractory_period
            return 1
        return 0