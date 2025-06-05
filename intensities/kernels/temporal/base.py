from abc import ABC, abstractmethod

class BaseTimeKernel(ABC):
    @abstractmethod
    def evaluate(self, dt: float) -> float:
        """
        Evaluate the temporal part of the kernel given the time difference dt.
        """
        pass
