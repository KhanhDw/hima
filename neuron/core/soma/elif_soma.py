import math
from .base_soma import BaseSoma

class ELIFSoma(BaseSoma):
    def __init__(self, E_L=-65.0, V_T=-55.0, Delta_T=2.0, g_L=0.1, V_peak=0.0, refractory_period=2):
        super().__init__(refractory_period)
        self.E_L = E_L
        self.V = E_L
        self.V_T = V_T
        self.Delta_T = Delta_T
        self.g_L = g_L
        self.V_peak = V_peak

    def step(self, current):
        if self.refractory_left > 0:
            self.refractory_left -= 1
            return 0

        exp_term = self.g_L * self.Delta_T * math.exp(min((self.V - self.V_T) / self.Delta_T, 10))
        dV = -self.g_L * (self.V - self.E_L) + exp_term + current
        self.V += dV * 0.1

        if self.V >= self.V_peak:
            self.V = self.E_L
            self.refractory_left = self.refractory_period
            return 1
        return 0