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
    def __init__(self, DATA_PATH,scaledOsimName, resultsDirectory):
        super(ImuInverseKinematicXMLProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH

        self.m_resultsDir = "" if resultsDirectory is None else resultsDirectory

        files.createDir(self.m_DATA_PATH+self.m_resultsDir) # required to save the mot file. (opensim issue ?) 

        self.m_osimName = DATA_PATH + scaledOsimName

        self.m_accuracy = 1e-8

    def setSetupFile(self,imuInverseKinematicToolFile):

        self.m_imuInverseKinematicTool = self.m_DATA_PATH + "__imuInverseKinematics_Setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(imuInverseKinematicToolFile,self.m_imuInverseKinematicTool)
        

    # def prepareDynamicTrial(self, acq, dynamicFile):

    #     self.m_dynamicFile = dynamicFile
    #     self.m_acq0 = acq
    #     self.m_acqMotion_forIK = btk.btkAcquisition.Clone(acq)

    #     self.m_ff = self.m_acqMotion_forIK.GetFirstFrame()
    #     self.m_freq = self.m_acqMotion_forIK.GetPointFrequency()

    #     opensimTools.transformMarker_ToOsimReferencial(self.m_acqMotion_forIK,self.m_progressionAxis,self.m_forwardProgression)

    #     self.m_markerFile = btkTools.smartWriter(self.m_acqMotion_forIK,self.m_DATA_PATH +  dynamicFile, extension="trc")

    #     self.m_beginTime = 0
    #     self.m_endTime = (self.m_acqMotion_forIK.GetLastFrame() - self.m_ff)/self.m_freq
    #     self.m_frameRange = [int((self.m_beginTime*self.m_freq)+self.m_ff),int((self.m_endTime*self.m_freq)+self.m_ff)] 



    # def setTimeRange(self,beginFrame=None,lastFrame=None):
    #     self.m_beginTime = 0.0 if beginFrame is None else (beginFrame-self.m_ff)/self.m_freq
    #     self.m_endTime = (self.m_acqMotion_forIK.GetLastFrame() - self.m_ff)/self.m_freq  if lastFrame is  None else (lastFrame-self.m_ff)/self.m_freq
    #     self.m_frameRange = [int((self.m_beginTime*self.m_freq)+self.m_ff),int((self.m_endTime*self.m_freq)+self.m_ff)]


    def prepareOrientationFile(self,imuMapper):
        self.m_imuMapper = imuMapper
        self.m_dynamicFile = "walking_orientations.sto"

        imuStorage = opensimIO.ImuStorageFile(self.m_DATA_PATH, "walking_orientations.sto")
        for key in self.m_imuMapper:
            imuStorage.setData(key,self.m_imuMapper[key].getQuaternions())
        imuStorage.construct(static=False)

        self.m_sensorOrientationFile = "walking_orientations.sto"

        opensimDf = opensimIO.OpensimDataFrame(self.m_DATA_PATH,"walking_orientations.sto")
        self.m_beginTime = opensimDf.m_dataframe["time"].iloc[0]
        self.m_endTime = opensimDf.m_dataframe["time"].iloc[-1]


    def setTimeRange(self,beginTime=None,lastTime=None):
        if beginTime is not None and beginTime <-self.m_endTime: 
            self.m_beginTime = beginTime
        if lastTime is not None and lastTime <-self.m_endTime: 
            self.m_endTime = lastTime


    def prepareXml(self):
        
        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("output_motion_file", self.m_DATA_PATH+self.m_resultsDir + "\\"+ self.m_dynamicFile+".mot")
        # for marker in self.m_weights.keys():
        #     self.xml.set_inList_fromAttr("IKMarkerTask","weight","name",marker,str(self.m_weights[marker]))
        # self.xml.set_one("accuracy",str(self.m_accuracy))
        self.xml.set_one("time_range",str(self.m_beginTime) + " " + str(self.m_endTime))
        self.xml.set_one("results_directory",  self.m_resultsDir)


    def run(self):

        self.xml.update()
        
        imu_ikTool = opensim.IMUInverseKinematicsTool(self.m_imuInverseKinematicTool)
        imu_ikTool.run()

    def finalize(self):
        # rename the xml setup file with the filename as suffix
        files.renameFile(self.m_imuInverseKinematicTool, 
                    self.m_DATA_PATH + self.m_dynamicFile[:-3] + "-imuInverseKinematicTool-setup.xml")