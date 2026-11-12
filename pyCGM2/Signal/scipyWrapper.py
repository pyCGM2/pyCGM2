"""Wrappers ensuring compatibility across SciPy versions."""
import scipy.integrate as si

def safe_cumtrapz(y, x=None, dx=1.0, axis=-1, initial=0):
    """Cumulative trapezoidal integration compatible with SciPy < 1.14 (``cumtrapz``) and >= 1.14 (``cumulative_trapezoid``).

    Args:
        y (array_like): Input array to integrate.
        x (array_like, optional): Sample points corresponding to the y values.
        dx (float, optional): Spacing between sample points when x is None. Defaults to 1.0.
        axis (int, optional): Axis along which to integrate. Defaults to -1.
        initial (float, optional): Value placed at the beginning of the result
            (same behaviour as the former ``cumtrapz(initial=0)``). Defaults to 0.

    Returns:
        np.ndarray: Cumulative integral of y along the given axis.
    """
    if hasattr(si, "cumulative_trapezoid"):
        # SciPy >= 1.14
        return si.cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=initial)
    else:
        # SciPy < 1.14
        return si.cumtrapz(y, x=x, dx=dx, axis=axis, initial=initial)