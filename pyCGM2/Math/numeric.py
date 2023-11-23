import numpy as np
from typing import List, Tuple, Dict, Optional,Union

def rms(x:np.ndarray, axis:Optional[int]=None)->np.ndarray:
    """
    Calculate the rms of an array

    Args:
        x (np.ndarray): array
        axis (int): direction of computation (0: columns, 1: rows)

    Returns:
        np.ndarray: rms array
        
    """

    return np.sqrt(np.mean(x**2, axis=axis))


def skewMatrix(vector:Union[np.ndarray,np.matrix]):
    """
    return a skew matrix from a vector

    Args:
        vector (Union[np.ndarray,np.matrix]) : array or matrix


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
