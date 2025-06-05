import math
import numpy as np
from .base import BaseSpaceKernel

class GaussianSpaceKernel(BaseSpaceKernel):
    """
    Spatial kernel using a Gaussian (RBF) function:
    exp( - (dx^2+dy^2)/(2*sigma^2) ) / (2*pi*sigma^2).
    """
    def __init__(self, sigma: float):
        self.sigma = sigma

    def evaluate(self, dx: float, dy: float) -> float:
        norm_sq = dx**2 + dy**2
        return np.exp(-norm_sq / (2 * self.sigma**2)) / (2 * math.pi * self.sigma**2)
