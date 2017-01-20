# -*- coding: utf-8 -*-
import numpy as np
import pdb

def timeSequenceNormalisation(Nrow,data):
    """ 
        Normalisation of an array

        :parameters:
            - `Nrow` (double) : number of interval
            - `data` (numpy.array(m,n)) : number of interval
            
    """
    
    ncol = data.shape[1]    
    out=np.zeros((Nrow,ncol))
    for i in range(0,ncol):
        out[:,i] = np.interp(np.linspace(0, 100, Nrow), np.linspace(0, 100, data.shape[0]), data[:,i]) 
    
    return out