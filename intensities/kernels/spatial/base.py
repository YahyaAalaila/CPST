from abc import ABC, abstractmethod

class BaseSpaceKernel(ABC):
    @abstractmethod
    def evaluate(self, dx: float, dy: float) -> float:
        """
        Evaluate the spatial part of the kernel given spatial differences dx, dy.
        """
        pass
