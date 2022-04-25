# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Model
#APIDOC["Draft"]=False
#--end--
"""
This module contains pose algorithms
"""

import numpy as np
import scipy as sp
import pyCGM2; LOGGER = pyCGM2.LOGGER

def segmentalLeastSquare(A, B):
    """Compute the transformation between two coordinate systems using SVD.

    Args:
        A (numpy.array(n,3)) - Coordinates [x,y,z] of at least three markers
        B (numpy.array(n,3)) - Coordinates [x,y,z] of at least three markers

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
