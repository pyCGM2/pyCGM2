from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import List, Tuple, Dict, Optional,Union

class Imu(object):
    """
    The IMU class represents an Inertial Measurement Unit (IMU) sensor.

    Args:
        freq (int): Sampling frequency of the IMU data.
        accel (Optional[np.ndarray]): Acceleration data from the IMU, represented as a NumPy array.
        angularVelocity (Optional[np.ndarray]): Angular velocity data from the IMU, represented as a NumPy array.
        mag (Optional[np.ndarray]): Magnetometer data from the IMU, represented as a NumPy array.
    """

 
    def __init__(self,freq:int,accel:Optional[np.ndarray],angularVelocity:Optional[np.ndarray],mag:Optional[np.ndarray]):
        """
        Initializes the Imu class with the specified frequency, acceleration data, angular velocity data, and magnetometer data.
        """
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
        """Restore initial values of IMU sensor data."""
        self.m_freq =  self._freq
        self.m_acceleration = self._accel
        self.m_angularVelocity = self._angularVelocity
        self.m_magnetometer = self._mag

    def update(self,newAccelValues:np.ndarray,newOmegaValues:np.ndarray,newMagValues:np.ndarray):
        """
        Update IMU with new data.

        Args:
            newAccelValues (np.ndarray): New acceleration data.
            newOmegaValues (np.ndarray): New angular velocity data.
            newMagValues (np.ndarray): New magnetometer data.
        """
        
        self.m_acceleration = newAccelValues
        self.m_angularVelocity =newOmegaValues
        self.m_magnetometer = newMagValues


    def downsample(self,freq:int=400):
        """
        Downsample IMU data to a specified frequency.

        Args:
            freq (int, optional): Target frequency for downsampling. Defaults to 400 Hz.
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
        """
        Returns acceleration data.

        Args:
            axis (Optional[str], optional): Specific axis ('X', 'Y', or 'Z') for which acceleration data is required. Defaults to None.

        Returns:
            np.ndarray: Acceleration data for the specified axis or all axes.
        """
        if axis is None: return self.m_acceleration
        elif axis=="X": return self.m_acceleration[:,0]
        elif axis=="Y": return self.m_acceleration[:,1]
        elif axis=="Z": return self.m_acceleration[:,2]
        else: 
            raise("[pyCGM2] - axis not known, choice: X,Y or Z")
        
    
    def getAngularVelocity(self,axis=None):
        """
        Returns angular velocity data.

        Args:
            axis (Optional[str], optional): Specific axis ('X', 'Y', or 'Z') for which angular velocity data is required. Defaults to None.

        Returns:
            np.ndarray: Angular velocity data for the specified axis or all axes.
        """
        if axis is None: return self.m_angularVelocity
        elif axis=="X": return self.m_angularVelocity[:,0]
        elif axis=="Y": return self.m_angularVelocity[:,1]
        elif axis=="Z": return self.m_angularVelocity[:,2]
        else: 
            raise("[pyCGM2] - axis not known, choice: X,Y or Z")
    
    def getMagnetometer(self,axis=None):
        """
        Returns magnetometer data.

        Args:
            axis (Optional[str], optional): Specific axis ('X', 'Y', or 'Z') for which magnetometer data is required. Defaults to None.

        Returns:
            np.ndarray: Magnetometer data for the specified axis or all axes.
        """
        if axis is None: return self.m_magnetometer
        elif axis=="X": return self.m_magnetometer[:,0]
        elif axis=="Y": return self.m_magnetometer[:,1]
        elif axis=="Z": return self.m_magnetometer[:,2]
        else: 
            raise("[pyCGM2] - axis not known, choice: X,Y or Z")

    
    def getMotion(self,index:Optional[int]=None):
        """
        Returns motion data. If an index is provided, returns the motion data at the specified index.

        Args:
            index (Optional[int], optional): Index for which motion data is required. Defaults to None.

        Returns:
            List[Frame] or Frame: Motion data for all frames or a specific frame at the given index.
        """

        if index is None: 
            return self.m_motion
        else:
            return self.m_motion[index]

    def getAngleAxis(self,axis=None):
        """
        Returns angle axis (equivalent to global angle) data.

        Args:
            axis (Optional[str], optional): Specific axis ('X', 'Y', or 'Z') for which angle axis data is required. Defaults to None.

        Returns:
            np.ndarray: Angle axis data for the specified axis or all axes.
        """
        values =  np.array([self.m_motion[i].getAngleAxis() for i in range(len(self.m_motion))])

        if axis is None: return values
        elif axis=="X": return values[:,0]
        elif axis=="Y": return values[:,1]
        elif axis=="Z": return values[:,2]
        else: 
            raise("[pyCGM2] - axis not known, choice: X,Y or Z")

    def getQuaternions(self):
        """
        Returns quaternion data representing the orientation of the IMU.

        Returns:
            np.ndarray: Quaternion data for each frame.
        """
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

