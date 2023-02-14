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

    def __init__(self,freq):
        self.m_freq =  freq

        self.m_acceleration = dict()
        self.m_gyro = dict()
        self.m_mag = dict()
        self.m_globalAngles = dict()
        self.m_highG = dict()

        self.m_properties = dict()

    def setAcceleration(self,axis,values):
        self.m_acceleration[axis] = values

    def setGyro(self,axis,values):
        self.m_gyro[axis] = values

    def setMagnetometer(self,axis,values):
        self.m_mag[axis] = values

    def downsample(self,freq=400):

        for axis in self.m_acceleration:
            values = self.m_acceleration[axis]
            time = np.arange(0, values.shape[0]/self.m_freq, 1/self.m_freq)
            f = interp1d(time, values, fill_value="extrapolate")
            newTime = np.arange(0, values.shape[0]/self.m_freq, 1/freq)
            self.m_acceleration[axis] = f(newTime)

        for axis in self.m_gyro:
            values = self.m_gyro[axis]
            time = np.arange(0, values.shape[0]/self.m_freq, 1/self.m_freq)
            f = interp1d(time, values, fill_value="extrapolate")
            newTime = np.arange(0, values.shape[0]/self.m_freq, 1/freq)
            self.m_gyro[axis] = f(newTime)

        self.m_freq = freq
        frames = np.arange(0, self.m_acceleration["X"].shape[0])
        self.m_time = frames*1/self.m_freq


    def constructDataFrame(self):

        frames = np.arange(0, self.m_acceleration["X"].shape[0])

        data = { "Time": frames*1/self.m_freq,
                "Accel.X": self.m_acceleration["X"],
                "Accel.Y": self.m_acceleration["Y"],
                "Accel.Z": self.m_acceleration["Z"],
                "Gyro.X": self.m_gyro["X"],
                "Gyro.Y": self.m_gyro["Y"],
                "Gyro.Z": self.m_gyro["Z"]}

        self.dataframe = pd.DataFrame(data)


    def constructTimeseries(self):
        """construct a kinetictoolkit timeseries
        """
        frames = np.arange(0, self.m_acceleration["X"].shape[0])

        self.m_timeseries = timeseries.TimeSeries()
        self.m_timeseries.time = frames*1/self.m_freq
        self.m_timeseries.data["Accel.X"] = self.m_acceleration["X"]
        self.m_timeseries.data["Accel.Y"] = self.m_acceleration["Y"]
        self.m_timeseries.data["Accel.Z"] = self.m_acceleration["Z"]
        self.m_timeseries.data["Gyro.X"] = self.m_gyro["X"]
        self.m_timeseries.data["Gyro.Y"] = self.m_gyro["Y"]
        self.m_timeseries.data["Gyro.Z"] = self.m_gyro["Z"]


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
        
