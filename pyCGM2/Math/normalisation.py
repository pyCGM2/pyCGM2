import numpy as np


def timeSequenceNormalisation(Nrow:int, data:np.ndarray)->np.ndarray:
    """
    Normalisation of an array

    Args:
        Nrow (int): number of interval
        data (np.ndarray): number of interval

    Returns:
        np.ndarray : array with 100 rows

    

        
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
