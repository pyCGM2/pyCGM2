from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import List, Tuple, Dict, Optional,Union

class Imu(object):
    """the IMU class

    Args:
        freq (int): frequency
        accel (Optional[np.ndarray]): acceleration
        angularVelocity (Optional[np.ndarray]): anular velocity
        mag (Optional[np.ndarray]): magnetometer
    """

 
    def __init__(self,freq:int,accel:Optional[np.ndarray],angularVelocity:Optional[np.ndarray],mag:Optional[np.ndarray]):

        self.m_freq =  freq

        self.m_acceleration =  accel

        if angularVelocity is None:
            self.m_angularVelocity = np.zeros((accel.shape[0],3))
        else:
            self.m_angularVelocity = angularVelocity
        if mag is None:
            self.m_magnetometer = np.zeros((accel.shape[0],3))
        else:
            self.m_magnetometer = mag

        
        self._freq =  freq
        self._accel = self.m_acceleration
        self._angularVelocity = self.m_angularVelocity
        self._mag = self.m_magnetometer

        self.m_properties = {}
        self.m_motion= []

        self.m_data =  pd.DataFrame()

    def reInit(self):
        """ restore initial values"""
        self.m_freq =  self._freq
        self.m_acceleration = self._accel
        self.m_angularVelocity = self._angularVelocity
        self.m_magnetometer = self._mag

    def update(self,newAccelValues:np.ndarray,newOmegaValues:np.ndarray,newMagValues:np.ndarray):
        """udpate imu with new data

        Args:
            newAccelValues (np.ndarray): acceleration
            newOmegaValues (np.ndarray): angular velocity
            newMagValues (np.ndarray): magnetometer
        """
        
        self.m_acceleration = newAccelValues
        self.m_angularVelocity =newOmegaValues
        self.m_magnetometer = newMagValues


    def downsample(self,freq:int=400):
        """downsample data

        Args:
            freq (int, optional): frequency. Defaults to 400.
        """        

        time = np.arange(0, self.m_acceleration.shape[0]/self.m_freq, 1/self.m_freq)
        newTime = np.arange(0, self.m_acceleration.shape[0]/self.m_freq, 1/freq)

        accel = np.zeros((newTime.shape[0],3))
        for i in range(0,3):
            values = self.m_acceleration[:,i]
            f = interp1d(time, values, fill_value="extrapolate")
            accel[:,i] = f(newTime)
        self.m_acceleration = accel

        angularVelocity = np.zeros((newTime.shape[0],3))
        for i in range(0,3):
            values = self.m_angularVelocity[:,i]
            f = interp1d(time, values, fill_value="extrapolate")
            angularVelocity[:,i] = f(newTime)
        self.m_angularVelocity =  angularVelocity


        mag = np.zeros((newTime.shape[0],3))
        for i in range(0,3):
            values = self.m_magnetometer[:,i]
            f = interp1d(time, values, fill_value="extrapolate")
            mag[:,i] = f(newTime)
        self.m_magnetometer =  mag
        
        self.m_freq = freq
        frames = np.arange(0, self.m_acceleration.shape[0])
        self.m_time = frames*1/self.m_freq

    def getAcceleration(self,axis=None):
        """ return acceleration"""
        if axis is None: return self.m_acceleration
        elif axis=="X": return self.m_acceleration[:,0]
        elif axis=="Y": return self.m_acceleration[:,1]
        elif axis=="Z": return self.m_acceleration[:,2]
        else: 
            raise("[pyCGM2] - axis not known, choice: X,Y or Z")
        
    
    def getAngularVelocity(self,axis=None):
        """ return angular velocity"""
        if axis is None: return self.m_angularVelocity
        elif axis=="X": return self.m_angularVelocity[:,0]
        elif axis=="Y": return self.m_angularVelocity[:,1]
        elif axis=="Z": return self.m_angularVelocity[:,2]
        else: 
            raise("[pyCGM2] - axis not known, choice: X,Y or Z")
    
    def getMagnetometer(self,axis=None):
        """ return magnetometer"""
        if axis is None: return self.m_magnetometer
        elif axis=="X": return self.m_magnetometer[:,0]
        elif axis=="Y": return self.m_magnetometer[:,1]
        elif axis=="Z": return self.m_magnetometer[:,2]
        else: 
            raise("[pyCGM2] - axis not known, choice: X,Y or Z")

    
    def getMotion(self,index:Optional[int]=None):
        """return motion. if index given, the method returns the motion at the specific index

        Args:
            index (Optional[int], optional): index. Defaults to None.
        """

        if index is None: 
            return self.m_motion
        else:
            return self.m_motion[index]

    def getAngleAxis(self,axis=None):
        """ return angle axis ( eq. global angle)"""
        values =  np.array([self.m_motion[i].getAngleAxis() for i in range(len(self.m_motion))])

        if axis is None: return values
        elif axis=="X": return values[:,0]
        elif axis=="Y": return values[:,1]
        elif axis=="Z": return values[:,2]
        else: 
            raise("[pyCGM2] - axis not known, choice: X,Y or Z")

    def getQuaternions(self):
        """ return quaternions"""
        return np.array([self.m_motion[i].getQuaternion() for i in range(len(self.m_motion))])



    # def constructDataFrame(self):

    #     frames = np.arange(0, self.m_acceleration.shape[0])

    #     data = { "Time": frames*1/self.m_freq,
    #             "Accel.X": self.m_acceleration[:,0],
    #             "Accel.Y": self.m_acceleration[:,1],
    #             "Accel.Z": self.m_acceleration[:,2],
    #             "Gyro.X": self.m_angularVelocity[:,0],
    #             "Gyro.Y": self.m_angularVelocity[:,1],
    #             "Gyro.Z": self.m_angularVelocity[:,2],
    #             "Mag.X": self.m_magnetometer[:,0],
    #             "Mag.Y": self.m_magnetometer[:,1],
    #             "Mag.Z": self.m_magnetometer[:,2]}

    #     self.dataframe = pd.DataFrame(data)


    # def constructTimeseries(self):
    #     """construct a kinetictoolkit timeseries
    #     """
    #     frames = np.arange(0, self.m_acceleration.shape[0])

    #     self.m_timeseries = timeseries.TimeSeries()
    #     self.m_timeseries.time = frames*1/self.m_freq
    #     self.m_timeseries.data["Accel.X"] = self.m_acceleration[:,0]
    #     self.m_timeseries.data["Accel.Y"] = self.m_acceleration[:,1]
    #     self.m_timeseries.data["Accel.Z"] = self.m_acceleration[:,2]

    #     self.m_timeseries.data["Gyro.X"] = self.m_angularVelocity[:,0]
    #     self.m_timeseries.data["Gyro.Y"] = self.m_angularVelocity[:,1]
    #     self.m_timeseries.data["Gyro.Z"] = self.m_angularVelocity[:,2]

    #     self.m_timeseries.data["Mag.X"] = self.m_magnetometer[:,0]
    #     self.m_timeseries.data["Mag.Y"] = self.m_magnetometer[:,1]
    #     self.m_timeseries.data["Mag.Z"] = self.m_magnetometer[:,2]

