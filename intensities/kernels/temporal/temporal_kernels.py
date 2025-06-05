import math
from .base import BaseTimeKernel

class ExponentialTimeKernel(BaseTimeKernel):
    """
    Temporal kernel using an exponential decay: beta * exp(-beta * dt).
    """
    def __init__(self, beta: float):
        self.beta = beta

    def evaluate(self, dt: float) -> float:
        if dt < 0:
            return 0.0
        return self.beta * math.exp(-self.beta * dt)
