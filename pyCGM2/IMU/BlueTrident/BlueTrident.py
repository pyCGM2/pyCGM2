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

    def __init__(self,freq):
        super(BlueTrident, self).__init__(freq)

        self.m_globalAngles = dict()
        self.m_highG = dict()

    def setGlobalAngles(self,axis,values):
        self.m_globalAngles[axis] = values
    
    def setHighAcceleration(self,axis,values):
        self.m_highG[axis] = values


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

        for axis in self.m_mag:
            values = self.m_mag[axis]
            time = np.arange(0, values.shape[0]/self.m_freq, 1/self.m_freq)
            f = interp1d(time, values, fill_value="extrapolate")
            newTime = np.arange(0, values.shape[0]/self.m_freq, 1/freq)
            self.m_mag[axis] = f(newTime)
        
        for axis in self.m_globalAngles:
            values = self.m_globalAngles[axis]
            time = np.arange(0, values.shape[0]/self.m_freq, 1/self.m_freq)
            f = interp1d(time, values, fill_value="extrapolate")
            newTime = np.arange(0, values.shape[0]/self.m_freq, 1/freq)
            self.m_globalAngles[axis] = f(newTime)
        
        for axis in self.m_highG:
            values = self.m_highG[axis]
            time = np.arange(0, values.shape[0]/self.m_freq, 1/self.m_freq)
            f = interp1d(time, values, fill_value="extrapolate")
            newTime = np.arange(0, values.shape[0]/self.m_freq, 1/freq)
            self.m_highG[axis] = f(newTime)

        self.m_freq = freq
        frames = np.arange(0, self.m_acceleration["X"].shape[0])
        self.m_time = frames*1/self.m_freq


    def getGlobalAngles(self):
        x = self.m_globalAngles["X"]
        y = self.m_globalAngles["Y"]
        z = self.m_globalAngles["Z"]

        return  np.array([x,y,z]).T    

    def getHighAcceleration(self):
        x = self.m_highG["X"]
        y = self.m_highG["Y"]
        z = self.m_highG["Z"]

        return  np.array([x,y,z]).T


    def constructDataFrame(self):

        frames = np.arange(0, self.m_acceleration["X"].shape[0])

        data = { "Time": frames*1/self.m_freq,
                "Accel.X": self.m_acceleration["X"],
                "Accel.Y": self.m_acceleration["Y"],
                "Accel.Z": self.m_acceleration["Z"],
                "Gyro.X": self.m_gyro["X"],
                "Gyro.Y": self.m_gyro["Y"],
                "Gyro.Z": self.m_gyro["Z"],
                "Mag.X": self.m_mag["X"],
                "Mag.Y": self.m_mag["Y"],
                "Mag.Z": self.m_mag["Z"],
                "GlobalAngle.X": self.m_globalAngles["X"],
                "GlobalAngle.Y": self.m_globalAngles["Y"],
                "GlobalAngle.Z": self.m_globalAngles["Z"],
                "HighG.X": self.m_highG["X"],
                "HighG.Y": self.m_highG["Y"],
                "HighG.Z": self.m_highG["Z"]}

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

        self.m_timeseries.data["Mag.X"] = self.m_mag["X"]
        self.m_timeseries.data["Mag.Y"] = self.m_mag["Y"]
        self.m_timeseries.data["Mag.Z"] = self.m_mag["Z"]

        self.m_timeseries.data["GlobalAngle.X"] = self.m_globalAngles["X"]
        self.m_timeseries.data["GlobalAngle.Y"] = self.m_globalAngles["Y"]
        self.m_timeseries.data["GlobalAngle.Z"] = self.m_globalAngles["Z"]

        self.m_timeseries.data["HighG.X"] = self.m_highG["X"]
        self.m_timeseries.data["HighG.Y"] = self.m_highG["Y"]
        self.m_timeseries.data["HighG.Z"] = self.m_highG["Z"]

    def computeOrientations(self, method = "ViconAngleAxis"):

        if method == "ViconAngleAxis":
        
            self.m_data["Orientations"]["ViconGlobalAngles"] =  dict()
            
            rotations=[]
            
            nAnalogFrames = self.m_globalAngles["X"].shape[0]

            quaternions = np.zeros((nAnalogFrames,4))
            
            eulerXYZ_angles = np.zeros((nAnalogFrames,3))
            eulerZYX_angles = np.zeros((nAnalogFrames,3))
            eulerXZY_angles = np.zeros((nAnalogFrames,3))
            eulerYZX_angles = np.zeros((nAnalogFrames,3))
            eulerYXZ_angles = np.zeros((nAnalogFrames,3))
            eulerZXY_angles = np.zeros((nAnalogFrames,3))

            globalAngles = self.getGlobalAngles()
            for i in range(0,nAnalogFrames): 
                rot = frame.getRotationMatrixFromAngleAxis(globalAngles[i,:])
                rotations.append(np.array(rot))

                quaternions[i,:] = frame.getQuaternionFromMatrix(rot)
                eulerXYZ_angles[i,:] = euler.euler_xyz(rot,similarOrder=False)
                eulerZYX_angles[i,:] = euler.euler_zyx(rot,similarOrder=False)
                eulerXZY_angles[i,:] = euler.euler_xzy(rot,similarOrder=False)
                eulerYZX_angles[i,:] = euler.euler_yzx(rot,similarOrder=False)
                eulerYXZ_angles[i,:] = euler.euler_yxz(rot,similarOrder=False)
                eulerZXY_angles[i,:] = euler.euler_zxy(rot,similarOrder=False)

                # rot = RotationMatrixFromAngleAxis( (self.m_globalAngles["X"][i],self.m_globalAngles["Y"][i], self.m_globalAngles["Z"][i])  )
                # rotations.append(np.array(rot))
                # quaternions[i,:] = np.array(QuaternionFromMatrix(rot))
                # eulerXYZ_angles[i,:] = np.array( EulerFromMatrix(rot, 'xyz'))
                # eulerZYX_angles[i,:] = np.array(EulerFromMatrix(rot, 'zyx'))
                # eulerXZY_angles[i,:] = np.array(EulerFromMatrix(rot, 'xzy'))
                # eulerYZX_angles[i,:] = np.array(EulerFromMatrix(rot, 'yzx'))
                # eulerYXZ_angles[i,:] = np.array(EulerFromMatrix(rot, 'yxz'))
                # eulerZXY_angles[i,:] = np.array(EulerFromMatrix(rot, 'zxy'))

            self.m_data["Orientations"]["ViconGlobalAngles"]["RotationMatrix"] = rot
            self.m_data["Orientations"]["ViconGlobalAngles"]["Quaternion"] = quaternions
            self.m_data["Orientations"]["ViconGlobalAngles"]["eulerXYZ"] = np.rad2deg(eulerXYZ_angles)
            self.m_data["Orientations"]["ViconGlobalAngles"]["eulerZYX"] = np.rad2deg(eulerZYX_angles)
            self.m_data["Orientations"]["ViconGlobalAngles"]["eulerXZY"] = np.rad2deg(eulerXZY_angles)
            self.m_data["Orientations"]["ViconGlobalAngles"]["eulerYZX"] = np.rad2deg(eulerYZX_angles)
            self.m_data["Orientations"]["ViconGlobalAngles"]["eulerYXZ"] = np.rad2deg(eulerYXZ_angles)
            self.m_data["Orientations"]["ViconGlobalAngles"]["eulerZXY"] = np.rad2deg(eulerZXY_angles)

 
 
        

    






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
        imuInstance = BlueTrident(freq)

        imuInstance.setAcceleration("X",data["ax_m/s/s"].to_numpy())
        imuInstance.setAcceleration("Y",data["ay_m/s/s"].to_numpy())
        imuInstance.setAcceleration("Z",data["az_m/s/s"].to_numpy())

        imuInstance.setGyro("X",data["gx_deg/s"].to_numpy())
        imuInstance.setGyro("Y",data["gy_deg/s"].to_numpy())
        imuInstance.setGyro("Z",data["gz_deg/s"].to_numpy())

        imuInstances.append(imuInstance)


    return imuInstances



def readBlueTridentCsv(fullfilename,freq):
    data = pd.read_csv(fullfilename)

    imuInstance = BlueTrident(freq)

    imuInstance.setAcceleration("X",data["ax_m/s/s"].to_numpy())
    imuInstance.setAcceleration("Y",data["ay_m/s/s"].to_numpy())
    imuInstance.setAcceleration("Z",data["az_m/s/s"].to_numpy())

    imuInstance.setGyro("X",data["gx_deg/s"].to_numpy())
    imuInstance.setGyro("Y",data["gy_deg/s"].to_numpy())
    imuInstance.setGyro("Z",data["gz_deg/s"].to_numpy())

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

    imuInstance = BlueTrident(acq.GetAnalogFrequency())

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

    for it in ["accel.x","accel.y","accel.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                     imuInstance.setAcceleration("X",channel.GetValues().reshape(nAnalogframes))
                if ".y" in it:
                    imuInstance.setAcceleration("Y",channel.GetValues().reshape(nAnalogframes))
                if ".z" in it:
                   imuInstance.setAcceleration("Z",channel.GetValues().reshape(nAnalogframes))

    for it in ["gyro.x","gyro.y","gyro.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                     imuInstance.setGyro("X",channel.GetValues().reshape(nAnalogframes))
                if ".y" in it:
                    imuInstance.setGyro("Y",channel.GetValues().reshape(nAnalogframes))
                if ".z" in it:
                   imuInstance.setGyro("Z",channel.GetValues().reshape(nAnalogframes))

    for it in ["mag.x","mag.y","mag.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                     imuInstance.setMagnetometer("X",channel.GetValues().reshape(nAnalogframes))
                if ".y" in it:
                    imuInstance.setMagnetometer("Y",channel.GetValues().reshape(nAnalogframes))
                if ".z" in it:
                   imuInstance.setMagnetometer("Z",channel.GetValues().reshape(nAnalogframes))


    for it in ["Global Angle.x","Global Angle.y","Global Angle.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                     imuInstance.setGlobalAngles("X",channel.GetValues().reshape(nAnalogframes))
                if ".y" in it:
                    imuInstance.setGlobalAngles("Y",channel.GetValues().reshape(nAnalogframes))
                if ".z" in it:
                   imuInstance.setGlobalAngles("Z",channel.GetValues().reshape(nAnalogframes))
    
    for it in ["HighG.x","HighG.y","HighG.z"]:
        for channel in channels:
            if it in channel.GetLabel():
                if ".x" in it:
                     imuInstance.setHighAcceleration("X",channel.GetValues().reshape(nAnalogframes))
                if ".y" in it:
                    imuInstance.setHighAcceleration("Y",channel.GetValues().reshape(nAnalogframes))
                if ".z" in it:
                   imuInstance.setHighAcceleration("Z",channel.GetValues().reshape(nAnalogframes))

    return imuInstance
