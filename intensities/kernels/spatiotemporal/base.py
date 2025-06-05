from abc import ABC, abstractmethod

class JointKernel(ABC):
    """
    Abstract base for joint spatiotemporal kernels, which take a joint input.
    """
    @abstractmethod
    def evaluate(self, dt: float, dx: float, dy: float) -> float:
        pass