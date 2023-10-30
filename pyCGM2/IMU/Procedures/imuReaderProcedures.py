import pandas as pd
import numpy as np
import btk
import re

import pyCGM2; LOGGER = pyCGM2.LOGGER  
from pyCGM2.IMU import imu
from pyCGM2.IMU import imuFilters
from pyCGM2.IMU.Procedures import imuMotionProcedure
from pyCGM2.Tools import btkTools



class ImuReaderProcedures(object):
    def __init__(self):
        pass

class CsvProcedure(ImuReaderProcedures):
    def __init__(self,fullfilename,translators):
        super(CsvProcedure, self).__init__()
        
        self.m_data = pd.read_csv(fullfilename)
        self.m_translators = translators



    def read(self):
    
        freq = int(1/(self.m_data["time_s"][1] - 
                      self.m_data["time_s"][0]))

        acceleration = np.array([self.m_data[self.m_translators["Accel.X"]].to_numpy(), 
                                self.m_data[self.m_translators["Accel.Y"]].to_numpy(), 
                                self.m_data[self.m_translators["Accel.Z"]].to_numpy()]).T

        angularVelocity = np.array([self.m_data[self.m_translators["AngularVelocity.X"]].to_numpy(), 
                                self.m_data[self.m_translators["AngularVelocity.Y"]].to_numpy(), 
                                self.m_data[self.m_translators["AngularVelocity.Z"]].to_numpy()]).T


        magnetometer = np.array([self.m_data[self.m_translators["Magneto.X"]].to_numpy(), 
                                self.m_data[self.m_translators["Magneto.Y"]].to_numpy(), 
                                self.m_data[self.m_translators["Magneto.Z"]].to_numpy()]).T


        imuInstance =  imu.Imu(freq,acceleration,angularVelocity,angularVelocity)

        
        requiredCols = [self.m_translators["Quaternion.X"],self.m_translators["Quaternion.Y"], self.m_translators["Quaternion.Z"], self.m_translators["Quaternion.R"]]
        if all(column in self.m_data.columns for column in requiredCols):
            LOGGER.logger.info("[pyCGM2] - your csv contains quaternions- - the reader compute the imu Motion")

            quaternions =  np.array([self.m_data[self.m_translators["Quaternion.X"]].to_numpy(),
                                    self.m_data[self.m_translators["Quaternion.Y"]].to_numpy(),
                                    self.m_data[self.m_translators["Quaternion.Z"]].to_numpy(), 
                                    self.m_data[self.m_translators["Quaternion.R"]].to_numpy()]).T

            motProc = imuMotionProcedure.QuaternionMotionProcedure(quaternions)

            motFilter = imuFilters.ImuMotionFilter(imuInstance,motProc)
            motFilter.run()
    
        return imuInstance
        


class C3dBlueTridentProcedure(ImuReaderProcedures):
    def __init__(self,fullfilename,viconDeviceId):
        super(C3dBlueTridentProcedure, self).__init__()
        
        self.m_acq = btkTools.smartReader(fullfilename)

        self.m_id = viconDeviceId


    def read(self):

        freq = self.m_acq.GetAnalogFrequency()
    
        channels = []
        for it in btk.Iterate(self.m_acq.GetAnalogs()):
            desc = it.GetDescription()
            label = it.GetLabel()
            values= it.GetValues()
            nAnalogframes = values.shape[0]

            if "Vicon BlueTrident" in desc:
                deviceId = re.findall( "\[(.*?)\]", desc)[0].split(",")[0]
                if deviceId ==  self.m_id:
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

        angularVelocity = np.zeros((nAnalogframes,3))
        for it in ["gyro.x","gyro.y","gyro.z"]:
            for channel in channels:
                if it in channel.GetLabel():
                    if ".x" in it:
                        angularVelocity[:,0] = channel.GetValues().reshape(nAnalogframes)
                    if ".y" in it:
                        angularVelocity[:,1] = channel.GetValues().reshape(nAnalogframes)
                    if ".z" in it:
                        angularVelocity[:,2] = channel.GetValues().reshape(nAnalogframes)
    
        magnetometer = np.zeros((nAnalogframes,3))
        for it in ["mag.x","mag.y","mag.z"]:
            for channel in channels:
                if it in channel.GetLabel():
                    if ".x" in it:
                        magnetometer[:,0] = channel.GetValues().reshape(nAnalogframes)
                    if ".y" in it:
                        magnetometer[:,1] = channel.GetValues().reshape(nAnalogframes)
                    if ".z" in it:
                        magnetometer[:,2] = channel.GetValues().reshape(nAnalogframes)


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
        
        imuInstance =  imu.Imu(freq,acceleration,angularVelocity,magnetometer)

        if np.all(globalAngle!=0):
            LOGGER.logger.info("[pyCGM2] - your c3d contains global Angle - the reader computes the imu Motion")

            motProc = imuMotionProcedure.GlobalAngleMotionProcedure(globalAngle)
            motFilter = imuFilters.ImuMotionFilter(imuInstance,motProc)
            motFilter.run()


        return imuInstance 
            
        

