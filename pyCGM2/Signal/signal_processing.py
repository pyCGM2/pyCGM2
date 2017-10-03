# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal, integrate

import btk



# ---- EMG -----

def remove50hz(array,fa):
    """
        Remove 50Hz signal
        
        :Parameters:
            - `array` (numpy.array(n,n)) - array
            - `fa` (double) - sample frequency 
   """
    bEmgStop, aEMGStop = signal.butter(2, np.array([49.9, 50.1]) / ((fa*0.5)), 'bandstop')
    value= signal.filtfilt(bEmgStop, aEMGStop, array,axis=0  ) 

    return value

def highPass(array,lowerFreq,upperFreq,fa):
    """
        High pass filtering
        
        :Parameters:
            - `array` (numpy.array(n,n)) - array
            - `lowerFreq` (double) - lower frequency 
            - `upperFreq` (double) - upper frequency 
            - `fa` (double) - sample frequency 
   """    
    bEmgHighPass, aEmgHighPass = signal.butter(2, np.array([lowerFreq, upperFreq]) / ((fa*0.5)), 'bandpass')
    value = signal.filtfilt(bEmgHighPass, aEmgHighPass,array - np.mean(array),axis=0  )
    
    return value

def rectify(array):
    """
        rectify a signal ( i.e get absolute values)
        
        :Parameters:
            - `array` (numpy.array(n,n)) - array

   """   
    return np.abs(array)
    
def enveloppe(array, fc,fa):
    """
        Get signal enveloppe from a low pass filter
        
        :Parameters:
            - `array` (numpy.array(n,n)) - array
            - `fc` (double) - cut-off frequency 
            - `fa` (double) - sample frequency 
   """   
    bEmgEnv, aEMGEnv = signal.butter(2, fc / (fa*0.5) , btype='lowpass')                       
    value = signal.filtfilt(bEmgEnv, aEMGEnv, array ,axis=0  )
    return value
    
    


# ---- btkAcq -----
def pointsFiltering(btkAcq,order=2, fc =6):
    """
        Low-pass filtering of all points in an acquisition 
        
        :Parameters:
            - `btkAcq` (btkAcquisition) - btk acquisition instance
            - `fc` (double) - cut-off frequency 
            - `order` (double) - order of the low-pass filter
   """   
    fp=btkAcq.GetPointFrequency()
    bPoint, aPoint = signal.butter(order, fc / (fp*0.5) , btype='lowpass')

    for pointIt in btk.Iterate(btkAcq.GetPoints()):
        x=signal.filtfilt(bPoint, aPoint, pointIt.GetValues()[:,0],axis=0  )
        y=signal.filtfilt(bPoint, aPoint, pointIt.GetValues()[:,1],axis=0  )
        z=signal.filtfilt(bPoint, aPoint, pointIt.GetValues()[:,2],axis=0  )
        pointIt.SetValues(np.array( [x,y,z] ).transpose())


# ----- methods ---------
def arrayLowPassFiltering(valuesArray, freq, order=2, fc =6):
    """
        low-pass filtering of an numpy array 
        
        :Parameters:
             - `valuesArray` (numpy.array(n,n)) - array
            - `fc` (double) - cut-off frequency
            - `order` (double) - order of the low-pass filter
    """
    b, a = signal.butter(order, fc / (freq*0.5) , btype='lowpass')

    out = np.zeros(valuesArray.shape)    
    for i in range(0, valuesArray.shape[1]):
        out[:,i] = signal.filtfilt(b, a, valuesArray[:,i] )
    
    return out
        


    