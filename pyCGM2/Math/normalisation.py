import numpy as np
from typing import List, Tuple, Dict, Optional,Union

def timeSequenceNormalisation(Nrow:int, data:np.ndarray)->np.ndarray:
    """
    Normalise a given data array to have a specified number of rows through linear interpolation.

    This function is useful for normalising time series data to a standard length,
    regardless of the original number of data points. It linearly interpolates
    the original data to fit into an array with the desired number of rows.

    Args:
        Nrow (int): Target number of rows for the normalised array.
        data (np.ndarray): Original data array to be normalised. Can be a 1D or 2D array.

    Returns:
        np.ndarray: Normalised array with 'Nrow' rows. If the input is a 1D array, the output will also be 1D.
    """
    
    if len(data.shape) == 1:
        flatArray = True
        data = data.reshape((data.shape[0], 1))
    else:
        flatArray = False

    ncol = data.shape[1]
    out = np.zeros((Nrow, ncol))
    for i in range(0, ncol):
        out[:, i] = np.interp(np.linspace(0, 100, Nrow),
                              np.linspace(0, 100, data.shape[0]), data[:, i])

    if flatArray:
        out = out.reshape(out.shape[0])

    return out
