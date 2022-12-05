from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Imu(object):
    """
    A node is a local position of a 3D point in a coordinate system

    Note:
        Automatically, the suffix "_node" ends the node label

    Args:
       label(str):  desired label of the node
       desc(str,Optional):  description
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
