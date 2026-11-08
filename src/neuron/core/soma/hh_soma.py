import math
from .base_soma import BaseSoma

class HodgkinHuxleySoma(BaseSoma):
    def __init__(self, E_K=-12.0, E_Na=115.0, E_L=10.613, g_K=36.0, g_Na=120.0, g_L=0.3, C_m=1.0):
        super().__init__(refractory_period=0)
        self.E_K = E_K
        self.E_Na = E_Na
        self.E_L = E_L
        self.g_K = g_K
        self.g_Na = g_Na
        self.g_L = g_L
        self.C_m = C_m

        self.V = 0.0
        self.m = 0.05
        self.h = 0.6
        self.n = 0.317
        self.was_above_threshold = False

    def step(self, current):
        alpha_m = (2.5 - 0.1 * self.V) / (math.exp(2.5 - 0.1 * self.V) - 1.0) if self.V != 25 else 1.0
        beta_m = 4.0 * math.exp(-self.V / 18.0)

        alpha_h = 0.07 * math.exp(-self.V / 20.0)
        beta_h = 1.0 / (math.exp(3.0 - 0.1 * self.V) + 1.0)

        alpha_n = (0.1 - 0.01 * self.V) / (math.exp(1.0 - 0.1 * self.V) - 1.0) if self.V != 10 else 0.1
        beta_n = 0.125 * math.exp(-self.V / 80.0)

        dt = 0.01
        for _ in range(10):
            self.m += dt * (alpha_m * (1 - self.m) - beta_m * self.m)
            self.h += dt * (alpha_h * (1 - self.h) - beta_h * self.h)
            self.n += dt * (alpha_n * (1 - self.n) - beta_n * self.n)

            I_Na = self.g_Na * (self.m ** 3) * self.h * (self.V - self.E_Na)
            I_K = self.g_K * (self.n ** 4) * (self.V - self.E_K)
            I_L = self.g_L * (self.V - self.E_L)

            dV = (current - I_Na - I_K - I_L) / self.C_m
            self.V += dV * dt

        spike = 0
        if self.V > 80.0 and not self.was_above_threshold:
            spike = 1
            self.was_above_threshold = True
        elif self.V <= 80.0:
            self.was_above_threshold = False

        return spike