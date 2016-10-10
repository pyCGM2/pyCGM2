# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:34:45 2016

@author: Fabien Leboeuf ( Salford Univ, UK)
"""
import numpy as np


def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))

def skewMatrix(vector):
    if isinstance(vector, ( np.matrix)):
        out=np.zeros((3,3))
        out[0,1]= -vector[0,2]    
        out[0,2]= vector[0,1]
    
        out[1,0]= vector[0,2]
        out[1,2]= -vector[0,0]
    
        out[2,0]= -vector[0,1]
        out[2,1]= vector[0,0]
        
        return np.matrix(out)
   
   
    elif isinstance(vector, ( np.ndarray)):

        out=np.zeros((3,3))
        out[0,1]= -vector[2]    
        out[0,2]= vector[1]
    
        out[1,0]= vector[2]
        out[1,2]= -vector[0]
    
        out[2,0]= -vector[1]
        out[2,1]= vector[0]
    
        return out