# stdgp/complexity/base_complexity.py
import abc

class BaseComplexity(abc.ABC):
    """
    Abstract base class for a complexity-measure module.
    """

    @abc.abstractmethod
    def measure(self, events, domain_info=None):
        """
        Compute a complexity measure given a set of events 
        and possibly domain info (extent of time/space, etc.).

        Return a float or dict with metrics.
        """
        pass
