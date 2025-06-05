from abc import ABC, abstractmethod

class BaseIntensity(ABC):
    """
    Unified base class for any intensity component.
    The evaluate() method returns a float intensity value.
    For background intensities, history is ignored;
    for triggering kernels, history is used.
    """
    @abstractmethod
    def evaluate(self, t: float, x: float, y: float, history: list = None) -> float:
        """
        Compute intensity at time t, location (x, y).
        
        Parameters:
            t: float - time
            x: float - spatial coordinate x
            y: float - spatial coordinate y
            history: list - (optional) list of past events
                     Each event is assumed to be a dict containing at least 't', 'x', 'y', 'type'
        
        Returns:
            float intensity value.
        """
        pass
