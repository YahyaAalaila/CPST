import numpy as np

def constant_value(c):
    """Return a function that always returns c."""
    return lambda t, x, y: c

def det_intensity(t, x, y):
    """An example deterministic function: a Gaussian bump in time and some spatial variation."""
    return np.exp(-((t - 6)**2) / 0.5) * (np.sin(x) + np.cos(y))

def polynomial_intensity(coeffs):
    """
    Return a function f(t, x, y) defined as a polynomial:
    f(t, x, y) = sum(coeffs[i] * (t, x, y)**exponents[i]).
    For simplicity, assume coeffs is a dict mapping (a, b, c) -> coefficient.
    """
    def f(t, x, y):
        val = 0.0
        for exponents, coeff in coeffs.items():
            a, b, c = exponents
            val += coeff * (t ** a) * (x ** b) * (y ** c)
        return val
    return f

# Registry mapping names to the corresponding functions.
VALUE_FUNCTION_REGISTRY = {
    "det_intensity": det_intensity,
    "polynomial_intensity": polynomial_intensity,
    # You can add a constant one dynamically, e.g., via constant_value, if needed.
}