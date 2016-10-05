# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 13:06:12 2016

@author: aaa34169
"""
import numpy as np
import pdb

def timeSequenceNormalisation(Nrow,data):
    """
    """
    
    ncol = data.shape[1]    
    out=np.zeros((Nrow,ncol))
    for i in range(0,ncol):
        out[:,i] = np.interp(np.linspace(0, 100, Nrow), np.linspace(0, 100, data.shape[0]), data[:,i]) 
    
    return out