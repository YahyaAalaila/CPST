from .base_intensity import BaseIntensity

class CompositeIntensity(BaseIntensity):
    """
    Overall intensity is the sum of background intensity and triggered (kernel) intensity.
    """
    def __init__(self, background: BaseIntensity, kernel: BaseIntensity):
        self.background = background
        self.kernel = kernel

    def evaluate(self, t: float, x: float, y: float, history: list = None) -> float:
        return self.background.evaluate(t, x, y, history) + self.kernel.evaluate(t, x, y, history)
