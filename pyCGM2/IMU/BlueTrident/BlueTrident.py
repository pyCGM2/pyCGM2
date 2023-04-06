# coding: utf-8
import re
import numpy as np
import pandas as pd

from pyCGM2.IMU import imu

import pyCGM2
LOGGER = pyCGM2.LOGGER

#from pyCGM2.Math.viconGeometry import RotationMatrixFromAngleAxis, QuaternionFromMatrix, EulerFromMatrix
from viconnexusapi.ViconUtils import RotationMatrixFromAngleAxis, QuaternionFromMatrix, EulerFromMatrix

from pyCGM2.Model import frame
from pyCGM2.Math import euler


from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from pyCGM2.External.ktk.kineticstoolkit import timeseries

try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")


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
        self.dataframe["GlobalAngle.X"]: self.m_globalAngles[:,0]
        self.dataframe["GlobalAngle.Y"]: self.m_globalAngles[:,1]
        self.dataframe["GlobalAngle.Z"]: self.m_globalAngles[:,2]
        self.dataframe["HighG.X"]: self.m_highG[:,0]
        self.dataframe["HighG.Y"]: self.m_highG[:,1]
        self.dataframe["HighG.Z"]: self.m_highG[:,2]



    def constructTimeseries(self):
        """construct a kinetictoolkit timeseries
        """
        super(BlueTrident, self).constructDataFrame()

        self.m_timeseries.data["GlobalAngle.X"] = self.m_globalAngles[:,0]
        self.m_timeseries.data["GlobalAngle.Y"] = self.m_globalAngles[:,1]
        self.m_timeseries.data["GlobalAngle.Z"] = self.m_globalAngles[:,2]

        self.m_timeseries.data["HighG.X"] = self.m_highG[:,0]
        self.m_timeseries.data["HighG.Y"] = self.m_highG[:,1]
        self.m_timeseries.data["HighG.Z"] = self.m_highG[:,2]

    def computeOrientations(self):
 
            rotations=[]
            nAnalogFrames = self.m_globalAngle.shape[0]

            quaternions = np.zeros((nAnalogFrames,4))
            
            # eulerXYZ_angles = np.zeros((nAnalogFrames,3))
            # eulerZYX_angles = np.zeros((nAnalogFrames,3))
            # eulerXZY_angles = np.zeros((nAnalogFrames,3))
            # eulerYZX_angles = np.zeros((nAnalogFrames,3))
            # eulerYXZ_angles = np.zeros((nAnalogFrames,3))
            # eulerZXY_angles = np.zeros((nAnalogFrames,3))

            globalAngles = self.getGlobalAngle()
            for i in range(0,nAnalogFrames): 
                rot = frame.getRotationMatrixFromAngleAxis(globalAngles[i,:])
                rotations.append(np.array(rot))

                quaternions[i,:] = frame.getQuaternionFromMatrix(rot)
                # eulerXYZ_angles[i,:] = euler.euler_xyz(rot,similarOrder=False)
                # eulerZYX_angles[i,:] = euler.euler_zyx(rot,similarOrder=False)
                # eulerXZY_angles[i,:] = euler.euler_xzy(rot,similarOrder=False)
                # eulerYZX_angles[i,:] = euler.euler_yzx(rot,similarOrder=False)
                # eulerYXZ_angles[i,:] = euler.euler_yxz(rot,similarOrder=False)
                # eulerZXY_angles[i,:] = euler.euler_zxy(rot,similarOrder=False)

            self.m_orientations["RotationMatrix"] = rotations
            self.m_orientations["Quaternion"] = quaternions


            # self.m_data["Orientations"]["ViconGlobalAngles"]["RotationMatrix"] = rotations
            # self.m_data["Orientations"]["ViconGlobalAngles"]["Quaternion"] = quaternions
            # self.m_data["Orientations"]["ViconGlobalAngles"]["eulerXYZ"] = np.rad2deg(eulerXYZ_angles)
            # self.m_data["Orientations"]["ViconGlobalAngles"]["eulerZYX"] = np.rad2deg(eulerZYX_angles)
            # self.m_data["Orientations"]["ViconGlobalAngles"]["eulerXZY"] = np.rad2deg(eulerXZY_angles)
            # self.m_data["Orientations"]["ViconGlobalAngles"]["eulerYZX"] = np.rad2deg(eulerYZX_angles)
            # self.m_data["Orientations"]["ViconGlobalAngles"]["eulerYXZ"] = np.rad2deg(eulerYXZ_angles)
            # self.m_data["Orientations"]["ViconGlobalAngles"]["eulerZXY"] = np.rad2deg(eulerZXY_angles)

 
    
    # def align(self, rotationmatrix):

    #     rotations=[]
    #     for i in range(0, len(self.m_orientations["RotationMatrix"] )
    #         rot = 
                       

        

    






def readmultipleBlueTridentCsv(fullfilenames,freq):

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx,array[idx]

    datasets= list()
    for fullfilename in fullfilenames:
        datasets.append( pd.read_csv(fullfilename))

    times0 = list()
    for dataset in datasets:
        times0.append(dataset["time_s"].values[0])
    time_0 =  max(times0)

    datasets_0 = list()
    numberOfFrames = list()
    for dataset in datasets:
        idx0,value0 = find_nearest(dataset["time_s"].values,time_0)
        numberOfFrames.append(dataset.iloc[idx0:].shape[0])
        datasets_0.append(dataset.iloc[idx0:])
    nFrames =  min(numberOfFrames)

    datasets_equal = list()
    for dataset in datasets_0:
        datasets_equal.append(dataset[0:nFrames])

    imuInstances = list()
    for data in datasets_equal:

        acceleration = np.array([data["ax_m/s/s"].to_numpy(), 
                                 data["ay_m/s/s"].to_numpy(), 
                                 data["az_m/s/s"].to_numpy()]).T
        
        angularVelocity = np.array([data["gx_deg/s"].to_numpy(), 
                            data["gy_deg/s"].to_numpy(), 
                            data["gz_deg/s"].to_numpy()]).T

        imuInstance = BlueTrident(freq,
                              acceleration,
                              angularVelocity,None,None,None)

        imuInstances.append(imuInstance)


    return imuInstances



def readBlueTridentCsv(fullfilename,freq):
    data = pd.read_csv(fullfilename)
    
    acceleration = np.array([data["ax_m/s/s"].to_numpy(), 
                             data["ay_m/s/s"].to_numpy(), 
                             data["az_m/s/s"].to_numpy()]).T

    angularVelocity = np.array([data["gx_deg/s"].to_numpy(), 
                             data["gy_deg/s"].to_numpy(), 
                             data["gz_deg/s"].to_numpy()]).T

    imuInstance = BlueTrident(freq,
                              acceleration,
                              angularVelocity,None,None,None)

    return imuInstance


def correctBlueTridentIds(acq):
    correctFlag = False

    # extract id from description
    ids = list()
    for it in btk.Iterate(acq.GetAnalogs()):
        desc = it.GetDescription()
        analoglabel = it.GetLabel()
        if "Vicon BlueTrident" in desc:
            id = re.findall( "\[(.*?)\]", desc)[0].split(",")

            if id[0] not in ids:  ids.append(id[0])

    if ids[0] !=1:
        correctFlag = True
        LOGGER.logger.warning("Blue trident ids do not start from 1 ")

    # correct description
    if correctFlag:
        newIds = list(np.arange(1,len(ids)+1))

        for it in btk.Iterate(acq.GetAnalogs()):
            desc = it.GetDescription()
            analoglabel = it.GetLabel()
            if "Vicon BlueTrident" in desc:
                id = re.findall( "\[(.*?)\]", desc)[0].split(",")[0]
                index = ids.index(id)
                newdesc = desc.replace("["+id, "["+str(newIds[index]))
                it.SetDescription(newdesc)

    return acq



def getBlueTrident(acq, id):

    

    channels = []
    for it in btk.Iterate(acq.GetAnalogs()):
        desc = it.GetDescription()
        label = it.GetLabel()
        values= it.GetValues()
        nAnalogframes = values.shape[0]

        if "Vicon BlueTrident" in desc:
            deviceId = re.findall( "\[(.*?)\]", desc)[0].split(",")[0]
            if deviceId == id:
                channels.append(it)

    acceleration = np.zeros((nAnalogframes,3))
    for it in ["accel.x","accel.y","accel.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                     acceleration[:,0] = channel.GetValues().reshape(nAnalogframes)
                if ".y" in it:
                    acceleration[:,1] = channel.GetValues().reshape(nAnalogframes)
                if ".z" in it:
                   acceleration[:,2] = channel.GetValues().reshape(nAnalogframes)

    omega = np.zeros((nAnalogframes,3))
    for it in ["gyro.x","gyro.y","gyro.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                    omega[:,0] = channel.GetValues().reshape(nAnalogframes)
                if ".y" in it:
                    omega[:,1] = channel.GetValues().reshape(nAnalogframes)
                if ".z" in it:
                   omega[:,2] = channel.GetValues().reshape(nAnalogframes)
 
    mag = np.zeros((nAnalogframes,3))
    for it in ["mag.x","mag.y","mag.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                     mag[:,0] = channel.GetValues().reshape(nAnalogframes)
                if ".y" in it:
                    mag[:,1] = channel.GetValues().reshape(nAnalogframes)
                if ".z" in it:
                   mag[:,2] = channel.GetValues().reshape(nAnalogframes)


    globalAngle = np.zeros((nAnalogframes,3))
    for it in ["Global Angle.x","Global Angle.y","Global Angle.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                    globalAngle[:,0] = channel.GetValues().reshape(nAnalogframes)
                if ".y" in it:
                    globalAngle[:,1] = channel.GetValues().reshape(nAnalogframes)
                if ".z" in it:
                   globalAngle[:,2] = channel.GetValues().reshape(nAnalogframes)

    highAccel = np.zeros((nAnalogframes,3))
    for it in ["HighG.x","HighG.y","HighG.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                    highAccel[:,0] = channel.GetValues().reshape(nAnalogframes)
                if ".y" in it:
                    highAccel[:,1] = channel.GetValues().reshape(nAnalogframes)
                if ".z" in it:
                   highAccel[:,2] = channel.GetValues().reshape(nAnalogframes)
    
    imuInstance = BlueTrident(acq.GetAnalogFrequency(),
                              acceleration,omega,mag,globalAngle,highAccel)
    return imuInstance 
