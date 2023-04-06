from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.linalg import norm

from pyCGM2.External.ktk.kineticstoolkit import timeseries
class Imu(object):
    """
    the Inertial meaure unit Class

    Args:
       freq(integer):  frequency
    """

    def __init__(self,freq,accel,angularVelocity,mag):

        self.m_freq =  freq

        self.m_accel =  accel
        if angularVelocity is None:
            self.m_angularVelocity = np.zeros((accel.shape[0],3))
        else:
            self.m_angularVelocity = angularVelocity
        
        if mag is None:
            self.m_mag = np.zeros((accel.shape[0],3))
        else:
            self.m_mag = mag

        self._accel = self.m_accel
        self._angularVelocity = self.m_angularVelocity
        self._mag = self.m_mag

        self.m_properties = dict()

        self.m_orientations= {"Method":None,
                              "RotationMatrix":None,
                              "Quaternions": None}

        self.m_state = "unaligned"

    def reInit(self):
        self.m_accel = self._accel
        self.m_angularVelocity = self._angularVelocity
        self.m_mag = self._mag

    def downsample(self,freq=400):

        time = np.arange(0, self.m_accel.shape[0]/self.m_freq, 1/self.m_freq)
        newTime = np.arange(0, self.m_accel.shape[0]/self.m_freq, 1/freq)

        accel = np.zeros((newTime.shape[0],3))
        for i in range(0,3):
            values = self.m_accel[:,i]
            f = interp1d(time, values, fill_value="extrapolate")
            accel[:,i] = f(newTime)
        self.m_accel = accel

        angularVelocity = np.zeros((newTime.shape[0],3))
        for i in range(0,3):
            values = self.m_angularVelocity[:,i]
            f = interp1d(time, values, fill_value="extrapolate")
            angularVelocity[:,i] = f(newTime)
        self.m_angularVelocity =  angularVelocity


        mag = np.zeros((newTime.shape[0],3))
        for i in range(0,3):
            values = self.m_mag[:,i]
            f = interp1d(time, values, fill_value="extrapolate")
            mag[:,i] = f(newTime)
        self.m_mag =  mag
        
        self.m_freq = freq
        frames = np.arange(0, self.m_accel.shape[0])
        self.m_time = frames*1/self.m_freq

    def getAcceleration(self):
        return self.m_accel
    
    def getAngularVelocity(self):
        return self.m_angularVelocity
    
    def getMagnetometer(self):
        return self.m_mag


    def constructDataFrame(self):

        frames = np.arange(0, self.m_accel.shape[0])

        data = { "Time": frames*1/self.m_freq,
                "Accel.X": self.m_accel[:,0],
                "Accel.Y": self.m_accel[:,1],
                "Accel.Z": self.m_accel[:,2],
                "Gyro.X": self.m_angularVelocity[:,0],
                "Gyro.Y": self.m_angularVelocity[:,1],
                "Gyro.Z": self.m_angularVelocity[:,2],
                "Mag.X": self.m_mag[:,0],
                "Mag.Y": self.m_mag[:,1],
                "Mag.Z": self.m_mag[:,2]}

        self.dataframe = pd.DataFrame(data)


    def constructTimeseries(self):
        """construct a kinetictoolkit timeseries
        """
        frames = np.arange(0, self.m_accel.shape[0])

        self.m_timeseries = timeseries.TimeSeries()
        self.m_timeseries.time = frames*1/self.m_freq
        self.m_timeseries.data["Accel.X"] = self.m_accel[:,0]
        self.m_timeseries.data["Accel.Y"] = self.m_accel[:,1]
        self.m_timeseries.data["Accel.Z"] = self.m_accel[:,2]

        self.m_timeseries.data["Gyro.X"] = self.m_angularVelocity[:,0]
        self.m_timeseries.data["Gyro.Y"] = self.m_angularVelocity[:,1]
        self.m_timeseries.data["Gyro.Z"] = self.m_angularVelocity[:,2]

        self.m_timeseries.data["Mag.X"] = self.m_mag[:,0]
        self.m_timeseries.data["Mag.Y"] = self.m_mag[:,1]
        self.m_timeseries.data["Mag.Z"] = self.m_mag[:,2]

    def getAccelerationPeaks(self, threshold=12, distance=50, plot= True):

        acceleration = norm(self.dataframe[["Accel.X","Accel.Y","Accel.Z"]],axis=1)  
        peaks, _ = find_peaks(acceleration,height=threshold, distance = distance)
        
        if plot:
            plt.plot(acceleration)
            plt.plot(peaks, acceleration[peaks], "x")
            plt.plot(np.zeros_like(acceleration), "--", color="gray")
            plt.show()
            
        data = { "Peaks": acceleration[peaks]}

        df = pd.DataFrame(data)

        #plt.figure()
        #df.Peaks.round(0).hist(orientation="horizontal")
        return df
        
