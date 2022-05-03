# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Math
#APIDOC["Draft"]=False
#--end--
import numpy as np


def timeSequenceNormalisation(Nrow, data):
    """
    Normalisation of an array

    Args:
        Nrow (double): number of interval
        data (array(m,n)): number of interval

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
