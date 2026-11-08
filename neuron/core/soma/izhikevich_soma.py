from .base_soma import BaseSoma

class IzhikevichSoma(BaseSoma):
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0):
        super().__init__(refractory_period=0)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.V = c
        self.u = b * self.V

    def step(self, current):
        dv = (0.04 * (self.V ** 2) + 5 * self.V + 140 - self.u + current)
        du = self.a * (self.b * self.V - self.u)

        self.V += dv * 0.5
        self.u += du * 0.5

        if self.V >= 30.0:
            self.V = self.c
            self.u += self.d
            return 1
        return 0