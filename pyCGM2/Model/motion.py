"""
This module contains pose algorithms
"""

import numpy as np
import scipy as sp
import pyCGM2; LOGGER = pyCGM2.LOGGER


from typing import List, Tuple, Dict, Optional

def segmentalLeastSquare(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Compute the transformation between two sets of coordinates using Singular Value Decomposition (SVD).

    This function calculates the rotation matrix, translation vector, root mean square error (RMSE), 
    and the centroids of both sets of coordinates. It uses SVD to optimize the least square problem
    for the transformation.

    Args:
        A (numpy.ndarray): An array of shape (n, 3) representing coordinates [x,y,z] of at least three markers.
        B (numpy.ndarray): An array of shape (n, 3) representing coordinates [x,y,z] of at least three markers.

    Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): The rotation matrix.
            - L (numpy.ndarray): The translation vector.
            - RMSE (float): The root mean square error.
            - Am (numpy.ndarray): The centroid of the first set of coordinates.
            - Bm (numpy.ndarray): The centroid of the second set of coordinates.
    """

    Am = np.mean(A, axis=0)           # centroid of m1
    Bm = np.mean(B, axis=0)           # centroid of m2
    M = np.dot((B - Bm).T, (A - Am))  # considering only rotation
    # singular value decomposition
    U, S, Vt = np.linalg.svd(M)
    # rotation matrix
    R = np.dot(U, np.dot(np.diag([1, 1, np.linalg.det(np.dot(U, Vt))]), Vt))
    # translation vector
    L = B.mean(0)  - np.dot(R, A.mean(0))
    # RMSE
    err = 0
    for i in range(A.shape[0]):
        Bp = np.dot(R, A[i, :]) + L
        err += np.sum((Bp - B[i, :])**2)
    try:
        RMSE = np.sqrt(err/A.shape[0]/3)
    except:
        RMSE =-1
        LOGGER.logger.warning("[pyCGM2] - residual of the least-square optimlization set to -1. gap presence ?")


    return R, L, RMSE, Am, Bm
