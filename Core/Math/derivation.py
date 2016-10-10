# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:21:57 2016

@author: Fabien Leboeuf ( Salford Univ, UK)
"""
import numpy as np
import scipy as sp
from scipy import interpolate

def splineFittingDerivation(values,sampleFrequency,order=1):
    N = values.shape[0]
    m = values.shape[1]
    
    x = np.linspace(0,N-1,N)
    
    out = np.zeros((N,m))    
    
    m=3
    smooth = m-np.sqrt(2*m)
    for i in range(0,m):
        spl = interpolate.splrep(x, values[:,i], k=5, s=smooth)
        der = interpolate.splev(x, spl, der=order)
        out[:,i] =  der    

    if order == 1 :   
        return out/ ((2*1/sampleFrequency))
    if order == 2 :   
        return out/ (np.power(1/sampleFrequency,2))



def splineDerivation(values,sampleFrequency,order=1):

    N = values.shape[0]
    m = values.shape[1]
    
    x = np.linspace(0,N-1,N)
    
    out = np.zeros((N,m))    
    
    for i in range(0,m):
        spl = interpolate.InterpolatedUnivariateSpline(x, values[:,i], k=5)
        der = spl.derivative(order)
        out[:,i] =  der(x)    
    if order == 1 :   
        return out/ ((2*1/sampleFrequency))
    if order == 2 :   
        return out/ (np.power(1/sampleFrequency,2))


def firstOrderFiniteDifference(values,sampleFrequency):
    
    n,m = values.shape
    out = np.zeros((n,m))
    
    i=0
    out[i,:] = (-3.0*values[i,:] +\
                    4.0*values[i+1,:] + \
                    -1.0*values[i+2,:]) / (2*1/sampleFrequency)
                    
    for i in range(1,n-1):
        postValue = values[i-1,:]
        nextValue = values[i+1,:]
        out[i,:]=( nextValue - postValue ) / (2*1/sampleFrequency);

    i=n-1
    out[i,:] = (3.0*values[i,:] +\
                    -4.0*values[i-1,:] + \
                    1.0*values[i-2,:]) / (2*1/sampleFrequency)
        
    return out



def secondOrderFiniteDifference(values,sampleFrequency):
    """
    derivateValues.row(i)=(1*this->m_Values.row(i-1) 
      -2*this->m_Values.row(i)
      +1*this->m_Values.row(i+1))/(pow(1/pointFrequency,2));
    """


    n,m = values.shape
    out = np.zeros((n,m))

    i=0
    out[i,:] = (-5.0*values[i+1,:] +\
                    4.0*values[i+2,:] + \
                    -1.0*values[i+3,:]) / (np.power(1/sampleFrequency,2))
                    
    for i in range(1,n-1):
        out[i,:] = (1.0*values[i-1,:] +\
                    -2.0*values[i,:] + \
                    1.0*values[i+1,:]) / (np.power(1/sampleFrequency,2))

    i=n-1
    out[i,:] = (-5.0*values[i-1,:] +\
                    4.0*values[i-2,:] + \
                    -1.0*values[i-3,:]) / (np.power(1/sampleFrequency,2))

    return out
   

def matrixFirstDerivation(motionList, sampleFrequency):

    nf = len(motionList)
    matrixList=[]

    #matrixList.append(np.zeros((3,3)))
    i=0
    matrixList.append( (-3.0*motionList[i].getRotation() + 4.0*motionList[i+1].getRotation()-motionList[i+2].getRotation())/(2*1/sampleFrequency))
 
    for i  in range(1,nf-1):
        matrixList.append( (motionList[i+1].getRotation()-motionList[i-1].getRotation())/(2*1/sampleFrequency))

    #matrixList.append(np.zeros((3,3)))
    i=nf-1        
    matrixList.append( (3.0*motionList[i].getRotation() - 4.0*motionList[i-1].getRotation() + motionList[i-2].getRotation())/(2*1/sampleFrequency))    
    return matrixList



def matrixSecondDerivation(motionList,sampleFrequency):
    nf = len(motionList)
    matrixList=[]

    #matrixList.append(np.zeros((3,3)))
    i=0
    matrixList.append( (-5.0*motionList[i+1].getRotation() + 4.0*motionList[i+2].getRotation()-motionList[i+3].getRotation())/(pow(1/sampleFrequency,2)))
    for i  in range(1,nf-1):
        matrixList.append( 
                    (motionList[i-1].getRotation()
                    -2.0*motionList[i].getRotation()
                    +1*motionList[i+1].getRotation() )/(pow(1/sampleFrequency,2))
                    )
    #matrixList.append(np.zeros((3,3)))
    i = nf-1                
    matrixList.append( (-5.0*motionList[i-1].getRotation() + 4.0*motionList[i-2].getRotation()-motionList[i-3].getRotation())/(pow(1/sampleFrequency,2)))    
    return matrixList