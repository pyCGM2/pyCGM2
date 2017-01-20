# -*- coding: utf-8 -*-

import numpy as np


def rms(x, axis=None):
    """
        Calculate the rms of an array
        
        :Parameters:
            - `x` (numpy.array(m,n)) : array
            - `axis` (int) direction of computation 
        
        .. note::
        
            - if axis=0 , you get rms for each column
            - if axis=1 , you get rms for each row
        
        :Return:
            - `na` (numpy.array(?,?))  array dimension depend on axis argument  
    """

    return np.sqrt(np.mean(x**2, axis=axis))

def skewMatrix(vector):
    """
        return a skew matrix from a vector
        
        :Parameters:
            - `vector` (numpy.array(3,)) : array

        :Return:
            - `na` (numpy.matrix) : skew matrix

    
    """    
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