# coding: utf-8
import numpy as np
import pandas as pd

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.IMU import imu
from pyCGM2.Model import frame
from pyCGM2.Math import euler


from scipy.interpolate import interp1d


class BlueTrident(imu.Imu):
    """
    a IMU-inherited class to work with vicon Blue Trident

    Args:
       freq(integer):  frequency
    """

    def __init__(self,freq,accel,angularVelocity,mag,globalAngle,highG):
        super(BlueTrident, self).__init__(freq,accel,angularVelocity,mag)

        if globalAngle is None:
            self.m_globalAngle = np.zeros((accel.shape[0],3))
        else:
            self.m_globalAngle = globalAngle
        
        if highG is None:
            self.m_highG = np.zeros((accel.shape[0],3))
        else:
            self.m_highG = highG

        self._highG = self.m_highG
        self._globalAngle = self.m_globalAngle
        

        self.m_orientations= {"Method":"From Vicon Global angle ",
                              "RotationMatrix":None,
                              "Quaternions": None}


    def reInit(self):
        super(BlueTrident, self).reInit()
        self.m_highG = self._highG
        self.m_globalAngle = self._globalAngle
        
   
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
        
        globalAngle = np.zeros((newTime.shape[0],3))
        for i in range(0,3):
            values = self.m_globalAngle[:,i]
            f = interp1d(time, values, fill_value="extrapolate")
            globalAngle[:,i] = f(newTime)
        self.m_globalAngle =  globalAngle
        
        highG = np.zeros((newTime.shape[0],3))
        for i in range(0,3):
            values = self.m_highG[:,i]
            f = interp1d(time, values, fill_value="extrapolate")
            highG[:,i] = f(newTime)
        self.m_highG =  highG
        
        self.m_freq = freq
        frames = np.arange(0, self.m_accel.shape[0])
        self.m_time = frames*1/self.m_freq
        

    def getGlobalAngle(self):
        return  self.m_globalAngle    

    def getHighAcceleration(self):
        return  self.m_highG


    def constructDataFrame(self):

        super(BlueTrident, self).constructDataFrame()
        self.dataframe["GlobalAngle.X"]: self.m_globalAngle[:,0]
        self.dataframe["GlobalAngle.Y"]: self.m_globalAngle[:,1]
        self.dataframe["GlobalAngle.Z"]: self.m_globalAngle[:,2]
        self.dataframe["HighG.X"]: self.m_highG[:,0]
        self.dataframe["HighG.Y"]: self.m_highG[:,1]
        self.dataframe["HighG.Z"]: self.m_highG[:,2]



    def constructTimeseries(self):
        """construct a kinetictoolkit timeseries
        """
        super(BlueTrident, self).constructTimeseries()

        self.m_timeseries.data["GlobalAngle.X"] = self.m_globalAngle[:,0]
        self.m_timeseries.data["GlobalAngle.Y"] = self.m_globalAngle[:,1]
        self.m_timeseries.data["GlobalAngle.Z"] = self.m_globalAngle[:,2]

        self.m_timeseries.data["HighG.X"] = self.m_highG[:,0]
        self.m_timeseries.data["HighG.Y"] = self.m_highG[:,1]
        self.m_timeseries.data["HighG.Z"] = self.m_highG[:,2]

    def computeOrientations(self):
 
            rotations=[]
            nAnalogFrames = self.m_globalAngle.shape[0]

            quaternions = np.zeros((nAnalogFrames,4))

            globalAngles = self.getGlobalAngle()
            for i in range(0,nAnalogFrames): 
                rot = frame.getRotationMatrixFromAngleAxis(globalAngles[i,:])
                rotations.append(np.array(rot))
                quaternions[i,:] = frame.getQuaternionFromMatrix(rot)

            self.m_orientations["RotationMatrix"] = rotations
            self.m_orientations["Quaternion"] = quaternions

    def computeAbsoluteAngles(self):
        nframes = len(self.m_orientations["RotationMatrix"])
        
        globalAngle = np.zeros((nframes,3))
        eulerXYZ_angles = np.zeros((nframes,3))
        eulerZYX_angles = np.zeros((nframes,3))
        eulerXZY_angles = np.zeros((nframes,3))
        eulerYZX_angles = np.zeros((nframes,3))
        eulerYXZ_angles = np.zeros((nframes,3))
        eulerZXY_angles = np.zeros((nframes,3))

        for i in range(0,nframes):
            rot = self.m_orientations["RotationMatrix"][i]
            quat = self.m_orientations["Quaternion"][i,:]

            globalAngle[i,0] = np.linalg.norm(frame.angleAxisFromQuaternion(quat,toRad=True))
            eulerXYZ_angles[i,:] = euler.euler_xyz(rot,similarOrder=False)
            eulerZYX_angles[i,:] = euler.euler_zyx(rot,similarOrder=False)
            eulerXZY_angles[i,:] = euler.euler_xzy(rot,similarOrder=False)
            eulerYZX_angles[i,:] = euler.euler_yzx(rot,similarOrder=False)
            eulerYXZ_angles[i,:] = euler.euler_yxz(rot,similarOrder=False)
            eulerZXY_angles[i,:] = euler.euler_zxy(rot,similarOrder=False)
        
        self.m_absoluteAngles["globalAngle"] = np.rad2deg(globalAngle)
        self.m_absoluteAngles["eulerXYZ"] = np.rad2deg(eulerXYZ_angles)
        self.m_absoluteAngles["eulerZYX"] = np.rad2deg(eulerZYX_angles)
        self.m_absoluteAngles["eulerXZY"] = np.rad2deg(eulerXZY_angles)
        self.m_absoluteAngles["eulerYZX"] = np.rad2deg(eulerYZX_angles)
        self.m_absoluteAngles["eulerYXZ"] = np.rad2deg(eulerYXZ_angles)
        self.m_absoluteAngles["eulerZXY"] = np.rad2deg(eulerZXY_angles)


    def align(self, blueTridentInstance):

        nframes = len(self.m_orientations["RotationMatrix"])
        r10 = blueTridentInstance.m_orientations["RotationMatrix"][0]

        quaternions = np.zeros((nframes,4))

        rotations=[]
        for i in range(0,nframes):
            r20 = self.m_orientations["RotationMatrix"][i]
            r12 = np.dot(r20.T, r10)
            r2f0 = np.dot(r20,r12)
            rotations.append(r2f0)
            quaternions[i,:] = frame.getQuaternionFromMatrix(r2f0)

        self.m_orientations["RotationMatrix"] = rotations
        self.m_orientations["Quaternion"] = quaternions

        for i in range(0,nframes):
            rot = self.m_orientations["RotationMatrix"][i]
            quat = self.m_orientations["Quaternion"][i,:]
            self.m_accel[i,:] = np.dot(rot,self.m_accel[i,:])
            self.m_angularVelocity[i,:] = np.dot(rot,self.m_angularVelocity[i,:])
            self.m_mag[i,:] = np.dot(rot,self.m_mag[i,:])
            self.m_highG[i,:] = np.dot(rot,self.m_highG[i,:])
            self.m_globalAngle[i,:] = frame.angleAxisFromQuaternion(quat, toRad=False)

        self._state = "aligned"
        


                       

        

    






