from abc import ABC, abstractmethod

class BaseParametricKernel(ABC):
    """
    Abstract base class for parametric kernels.
    
    This interface defines a method to compute the triggered (kernel) contribution
    from a past event at (t_i, x_i, y_i, past_type) to a candidate event at (t, x, y, candidate_type).
    """
    
    @abstractmethod
    def evaluate(self, t: float, x: float, y: float) -> float:
        """
        Evaluate the kernel contribution.
        
        Parameters:
            t, x, y: float – candidate event's time and spatial coordinates.
            candidate_type: int – candidate event's mark/type.
            t_i, x_i, y_i: float – past event's time and spatial coordinates.
            past_type: int – past event's mark/type.
        
        Returns:
            float – the kernel (triggered) contribution.
        """
        pass