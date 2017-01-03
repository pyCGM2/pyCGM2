# -*- coding: utf-8 -*-
"""
Created on Sat May 09 22:46:25 2015

@author: fleboeuf
"""

import os
import sys

import btk
import pdb
import numpy as np
import scipy as sp



def segmentalLeastSquare(A, B):
    """Calculates the transformation between two coordinate systems using SVD.

    See the help of the svdt function.

    Parameters
    ----------
    A   : 2D Numpy array (n, 3), where n is the number of markers.
        Coordinates [x,y,z] of at least three markers
    B   : 2D Numpy array (n, 3), where n is the number of markers.
        Coordinates [x,y,z] of at least three markers

    Returns
    -------
    R    : 2D Numpy array (3, 3)
         Rotation matrix between A and B
    L    : 1D Numpy array (3,)
         Translation vector between A and B
    RMSE : float
         Root-mean-squared error for the rigid body model: B = R*A + L + err.

    See Also
    --------
    numpy.linalg.svd
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
    RMSE = np.sqrt(err/A.shape[0]/3)


    return R, L, RMSE, Am, Bm