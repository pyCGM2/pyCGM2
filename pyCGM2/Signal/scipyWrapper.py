import scipy.integrate as si

def safe_cumtrapz(y, x=None, dx=1.0, axis=-1, initial=0):
    """
    Wrapper compatible entre SciPy <1.14 (cumtrapz) et >=1.14 (cumulative_trapezoid).

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the y values.
    dx : scalar, optional
        Spacing between sample points when x is None.
    axis : int, optional
        Axis along which to integrate.
    initial : scalar, optional
        If given, place this value at the beginning of the returned result.
        Behaves like old cumtrapz(initial=0) when initial=0.

    Returns
    -------
    ndarray
        Cumulative integral of y along the given axis.
    """
    if hasattr(si, "cumulative_trapezoid"):
        # SciPy >= 1.14
        return si.cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=initial)
    else:
        # SciPy < 1.14
        return si.cumtrapz(y, x=x, dx=dx, axis=axis, initial=initial)