from .base import JointKernel
import math
class PolynomialJointKernel(JointKernel):
    """
    A simple joint kernel modeled as a polynomial in dt, dx, dy.
    For example: k(dt, dx, dy) = c0 + c1*dt + c2*dx + c3*dy + c4*dt*dx + ...
    The coefficients can be tuned to control the joint behavior.
    """
    def __init__(self, coefficients: dict):
        """
        coefficients: dict where keys are tuples indicating exponents,
        e.g. (a, b, c) corresponding to dt^a * dx^b * dy^c, and values are coefficients.
        For example: {(0,0,0): 1.0, (1,0,0): 0.5, (0,1,0): -0.1}
        """
        self.coefficients = coefficients

    def evaluate(self, dt: float, dx: float, dy: float) -> float:
        val = 0.0
        for exponents, coeff in self.coefficients.items():
            a, b, c = exponents
            val += coeff * (dt ** a) * (dx ** b) * (dy ** c)
        return max(val, 0.0)