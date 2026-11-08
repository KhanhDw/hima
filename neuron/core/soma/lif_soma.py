from .base_soma import BaseSoma

class Somas(BaseSoma): # Hoặc giữ nguyên tên Soma cũ
    def __init__(
        self,
        theta=1.0,
        leak=0.9,
        refractory_period=2
    ):
        super().__init__(refractory_period)
        self.theta = theta
        self.leak = leak

    def step(self, current):
        if self.refractory_left > 0:
            self.refractory_left -= 1
            return 0

        self.V = (
            self.V * self.leak
            + current
        )

        if self.V >= self.theta:
            self.V = 0.0
            self.refractory_left = (
                self.refractory_period
            )
            return 1

        return 0