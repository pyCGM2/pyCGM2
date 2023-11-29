import numpy as np
from typing import List, Tuple, Dict, Optional,Union

def rms(x:np.ndarray, axis:Optional[int]=None)->np.ndarray:
    """
    Calculate the root mean square (RMS) along a specified axis of an array.

    The RMS is a measure of the magnitude of a set of numbers. It is the square root of the 
    arithmetic mean of the squares of the original values. This function is often used in 
    signal processing to quantify the variation of a signal.

    Args:
        x (np.ndarray): The input array for which the RMS is calculated.
        axis (Optional[int]): The axis along which to compute the RMS. If None, the RMS is computed
                              over the entire array. If axis is 0, RMS is computed column-wise. 
                              If axis is 1, RMS is computed row-wise.

    Returns:
        np.ndarray: An array containing the RMS values. The shape of the output depends on the input array 
                    and the axis parameter.
    """

    return np.sqrt(np.mean(x**2, axis=axis))


def skewMatrix(vector:Union[np.ndarray,np.matrix]):
    """
    Generate a skew-symmetric matrix from a given vector.

    A skew-symmetric (or antisymmetric) matrix is a square matrix whose transpose equals its negative.
    This function is commonly used in 3D mathematics, particularly in computing vector cross products,
    rigid body dynamics, and other applications in physics and engineering.

    Args:
        vector (Union[np.ndarray, np.matrix]): A 3-element vector from which the skew-symmetric matrix is generated.
                                               Can be either a numpy array or a numpy matrix.

    Returns:
        A 3x3 skew-symmetric matrix derived from the input vector.
    """
    if isinstance(vector, (np.matrix)):
        out = np.zeros((3, 3))
        out[0, 1] = -vector[0, 2]
        out[0, 2] = vector[0, 1]

        out[1, 0] = vector[0, 2]
        out[1, 2] = -vector[0, 0]

        out[2, 0] = -vector[0, 1]
        out[2, 1] = vector[0, 0]

        return np.matrix(out)

    elif isinstance(vector, (np.ndarray)):

        out = np.zeros((3, 3))
        out[0, 1] = -vector[2]
        out[0, 2] = vector[1]

        out[1, 0] = vector[2]
        out[1, 2] = -vector[0]

        out[2, 0] = -vector[1]
        out[2, 1] = vector[0]

        return out
