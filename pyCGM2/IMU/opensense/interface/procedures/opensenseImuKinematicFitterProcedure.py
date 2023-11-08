# -*- coding: utf-8 -*-
import os
from distutils import extension
from weakref import finalize
from pyCGM2.Utils import files
from pyCGM2.Tools import btkTools
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Model.Opensim import opensimIO

import btk
import opensim


class ImuInverseKinematicXMLProcedure(object):
    def __init__(self, DATA_PATH,calibratedOsimName, resultsDirectory):
        super(ImuInverseKinematicXMLProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH

        self.m_resultsDir = "" if resultsDirectory is None else resultsDirectory

        files.createDir(self.m_DATA_PATH+self.m_resultsDir) # required to save the mot file. (opensim issue ?) 

        self.m_osimName = DATA_PATH + calibratedOsimName
        self.m_osimInterface = opensimInterface.osimInterface(self.m_DATA_PATH, calibratedOsimName)

        self.m_accuracy = 1e-8
        self.m_imuMapper = dict()

        self.m_sensor_to_opensim_rotations = None

    def setSetupFile(self,imuInverseKinematicToolFile):

        self.m_imuInverseKinematicTool = self.m_DATA_PATH + "__imuInverseKinematics_Setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(imuInverseKinematicToolFile,self.m_imuInverseKinematicTool)

    def setImuMapper(self,imuMapperDict):
        for key in imuMapperDict:
            if key not in self.m_osimInterface.getBodies():
                LOGGER.logger.error(f"[pyCGM2] the key {key} of your mapper is not a body of the osim file")
                raise Exception (f"[pyCGM2] the key {key} of your mapper is not a body of the osim file") 
                
        self.m_imuMapper.update(imuMapperDict)

    def placeImu(self,osimBody,imuInstance):
        if osimBody not in self.m_osimInterface.getBodies():
            LOGGER.logger.error(f"[pyCGM2] the key {osimBody} of your mapper is not a body of the osim file")
            raise Exception (f"[pyCGM2] the key {osimBody} of your mapper is not a body of the osim file") 

        self.m_imuMapper.update({osimBody : imuInstance})

    def prepareOrientationFile(self,motionFilenameNoExt,freq,order=[0,1,2,3]):
        
        self.m_dynamicFile = motionFilenameNoExt+".sto"

        imuStorage = opensimIO.ImuStorageFile(self.m_DATA_PATH, self.m_dynamicFile,freq)
        for key in self.m_imuMapper:
            imuStorage.setData(key,self.m_imuMapper[key].getQuaternions()[:, order])
        imuStorage.construct(static=False)

        self.m_sensorOrientationFile = self.m_dynamicFile

        opensimDf = opensimIO.OpensimDataFrame(self.m_DATA_PATH,self.m_dynamicFile)
        self.m_beginTime = opensimDf.m_dataframe["time"].iloc[0]
        self.m_endTime = opensimDf.m_dataframe["time"].iloc[-1]


    def setTimeRange(self,beginTime=None,lastTime=None):
        if beginTime is not None and beginTime <-self.m_endTime: 
            self.m_beginTime = beginTime
        if lastTime is not None and lastTime <-self.m_endTime: 
            self.m_endTime = lastTime

    def setSensorToOpensimRotation(self,eulerAngles):
        if len(eulerAngles) !=3:
            raise ("[pyCGM2] - eulerAngles must be a list of 3 values in radian")
        self.m_sensor_to_opensim_rotations = eulerAngles

    def prepareXml(self):
        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("orientations_file", self.m_dynamicFile)


        self.xml.set_one("output_motion_file", self.m_DATA_PATH+self.m_resultsDir + "\\"+ self.m_dynamicFile[:-4]+".mot")
        # for marker in self.m_weights.keys():
        #     self.xml.set_inList_fromAttr("IKMarkerTask","weight","name",marker,str(self.m_weights[marker]))
        # self.xml.set_one("accuracy",str(self.m_accuracy))
        self.xml.set_one("time_range",str(self.m_beginTime) + " " + str(self.m_endTime))
        self.xml.set_one("results_directory",  self.m_resultsDir)

        if self.m_sensor_to_opensim_rotations is not None:
            self.xml.set_one("sensor_to_opensim_rotations", ' '.join(str(e) for e in self.m_sensor_to_opensim_rotations)) 



    def run(self):

        self.xml.update()
        
        imu_ikTool = opensim.IMUInverseKinematicsTool(self.m_imuInverseKinematicTool)
        imu_ikTool.run()

    def finalize(self):
        # rename the xml setup file with the filename as suffix
        files.renameFile(self.m_imuInverseKinematicTool, 
                    self.m_DATA_PATH + self.m_dynamicFile[:-3] + "-imuInverseKinematicTool-setup.xml")