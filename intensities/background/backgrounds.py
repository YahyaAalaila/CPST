from ..base_intensity import BaseIntensity
import numpy as np
from .value_functions import VALUE_FUNCTION_REGISTRY

class ConstantIntensity(BaseIntensity):
    """Constant background intensity: mu(t,x,y) = c."""
    def __init__(self, c: float):
        self.c = c

    def evaluate(self, t: float, x: float, y: float, history: list = None) -> float:
        return self.c


class DeterministicIntensity(BaseIntensity):
    """
    Deterministic background intensity:
      mu(t, x, y) = exp(a0 + a1*sin(omega * t)) * exp(-||[x,y] - s0||/sigma0)
    """
    def __init__(self, a0: float, a1: float, omega: float, s0: np.ndarray, sigma0: float):
        self.a0 = a0
        self.a1 = a1
        self.omega = omega
        self.s0 = s0
        self.sigma0 = sigma0

    def evaluate(self, t: float, x: float, y: float, history: list = None) -> float:
        distance = np.linalg.norm(np.array([x, y]) - self.s0)
        f_t = np.exp(self.a0 + self.a1 * np.sin(self.omega * t))
        g_s = np.exp(-distance / self.sigma0)
        return f_t * g_s


class PiecewiseIntensity(BaseIntensity):
    """
    Piecewise intensity function that partitions the time and spatial domain,
    and assigns an intensity to each partition. The intensity for a given partition
    can be specified as either:
      - A constant (float), or
      - A callable function f(t, x, y) that returns a float.
    
    This flexibility allows you to model the background intensity as constant in some regions,
    while using a polynomial or other deterministic function in others.
    """
    def __init__(self, time_partitions: list, space_partitions: list, values_dict: dict):
        """
        Parameters:
            time_partitions: list of floats, e.g. [0, 5, 10], partitioning time.
            space_partitions: list of region dicts, e.g.
                              [{'x_min': 0, 'x_max': 5, 'y_min': 0, 'y_max': 10}, ...]
                              which partition the spatial domain.
            values_dict: dict mapping (time_bin, space_bin) -> intensity,
                         where intensity is either a float or a callable f(t, x, y).
        """
        self.time_partitions = time_partitions
        self.space_partitions = space_partitions
        self.values_dict = {self._parse_key(k): v for k, v in values_dict.items()}

    def _parse_key(self, key):
        # If key is a string like "(1, 0)", convert it to a tuple (1, 0)
        if isinstance(key, str):
            return tuple(map(int, key.strip("()").split(",")))
        return key

    def evaluate(self, t: float, x: float, y: float, history: list = None) -> float:
        # Determine in which time and space bin the point (t, x, y) falls.
        time_bin = self._find_time_bin(t)
        space_bin = self._find_space_bin(x, y)
        # Retrieve the value for this partition.
        value = self.values_dict.get((time_bin, space_bin), 0.0)
        # If value is a dict and has a 'function' key, look up the corresponding callable.
        if isinstance(value, dict) and "function" in value:
            func_name = value["function"]
            func = VALUE_FUNCTION_REGISTRY.get(func_name)
            if func is None:
                raise ValueError(f"Function {func_name} not found in registry.")
            return func(t, x, y)
        elif callable(value):
            return value(t, x, y)
        else:
            return value

    def _find_time_bin(self, t: float) -> int:
        for i in range(len(self.time_partitions) - 1):
            if self.time_partitions[i] <= t < self.time_partitions[i+1]:
                return i
        return len(self.time_partitions) - 2

    def _find_space_bin(self, x: float, y: float) -> int:
        for j, region in enumerate(self.space_partitions):
            if (region['x_min'] <= x < region['x_max'] and
                region['y_min'] <= y < region['y_max']):
                return j
        return -1