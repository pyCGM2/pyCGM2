# -*- coding: utf-8 -*-
import os
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from bs4 import BeautifulSoup

# pyCGM2
try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")

from pyCGM2.Tools import  btkTools,opensimTools
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures
from pyCGM2.Utils import files
try:
    import opensim
except:
    try:
        from pyCGM2 import opensim4 as opensim
    except:
        LOGGER.logger.error("[pyCGM2] opensim not found on your system")
from pyCGM2.Model.Opensim import opensimIO


class InverseKinematicXmlProcedure(opensimProcedures.OpensimInterfaceXmlProcedure):
    def __init__(self,DATA_PATH,scaledOsimName, resultsDirectory):

        super(InverseKinematicXmlProcedure,self).__init__()
        self.m_DATA_PATH = DATA_PATH

        self.m_resultsDir = "" if resultsDirectory is None else resultsDirectory+"//"
        self.m_RES_PATH = self.m_DATA_PATH+self.m_resultsDir
        files.createDir(self.m_DATA_PATH+self.m_resultsDir) # required to save the mot file. (opensim issue ?) 

        self.m_osimName = DATA_PATH + scaledOsimName
        self.m_accuracy = 1e-8
        self.m_acq = None

    def setSetupFile(self, ikToolFile):
        self.m_ikTool = self.m_DATA_PATH + "__IKTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(ikToolFile,self.m_ikTool)
    
    def prepareTrial_fromBtkAcq(self, acq, dynamicFile,progressionAxis,forwardProgression):


        self.m_dynamicFile = dynamicFile
        self.m_acq0 = acq
        self.m_acqMotion_forIK = btk.btkAcquisition.Clone(acq)

        self.m_ff = self.m_acqMotion_forIK.GetFirstFrame()
        self.m_freq = self.m_acqMotion_forIK.GetPointFrequency()

        self.m_progressionAxis = progressionAxis
        self.m_forwardProgression = forwardProgression
        self.m_R_LAB_OSIM = opensimTools.rotationMatrix_labToOsim(progressionAxis,forwardProgression)


        opensimTools.transformMarker_ToOsimReferencial(self.m_acqMotion_forIK,self.m_progressionAxis,self.m_forwardProgression)

        self.m_markerFile = btkTools.smartWriter(self.m_acqMotion_forIK,self.m_DATA_PATH +  dynamicFile, extension="trc")

        self.m_beginTime = 0
        self.m_endTime = (self.m_acqMotion_forIK.GetLastFrame() - self.m_ff)/self.m_freq
        self.m_frameRange = [int((self.m_beginTime*self.m_freq)+self.m_ff),int((self.m_endTime*self.m_freq)+self.m_ff)] 

    def prepareWeights(self,weights_dict):
        self.m_weights = weights_dict

    def setAccuracy(self,value):
        self.m_accuracy = value

    def setFrameRange(self,begin,end):
        if self.m_acq is None:
            raise Exception(f"[pyCGM2] - no acquisition detected - trial preparation from a btk::Acquisition not done")
        self.m_beginTime = 0.0 if begin is None else (begin-self.m_ff)/self.m_freq
        self.m_endTime = (self.m_acq.GetLastFrame() - self.m_ff)/self.m_freq  if end is  None else (end-self.m_ff)/self.m_freq

    def setTimeRange(self,begin,end):
        self.m_beginTime = begin
        self.m_endTime = end

    def prepareXml(self):

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("marker_file", files.getFilename(self.m_markerFile))
        self.xml.set_one("output_motion_file", self.m_RES_PATH+ self.m_dynamicFile+".mot")
        for marker in self.m_weights.keys():
            self.xml.set_inList_fromAttr("IKMarkerTask","weight","name",marker,str(self.m_weights[marker]))
        self.xml.set_one("accuracy",str(self.m_accuracy))
        self.xml.set_one("time_range",str(self.m_beginTime) + " " + str(self.m_endTime))
        self.xml.set_one("results_directory",  self.m_resultsDir)

    def run(self):

        if os.path.isfile(self.m_DATA_PATH +self.m_dynamicFile+"_ik_model_marker_locations.sto"):
            os.remove(self.m_DATA_PATH +self.m_dynamicFile+"_ik_model_marker_locations.sto")
        if os.path.isfile(self.m_DATA_PATH +self.m_dynamicFile+"_ik_marker_errors.sto"):
            os.remove(self.m_DATA_PATH +self.m_dynamicFile+"_ik_marker_errors.sto")

        self.xml.update()

        if not hasattr(self, "m_frameRange"):
            time_range_str = self.xml.m_soup.find("time_range").string
            time_range = [float(it) for it in time_range_str.split(" ")]
            self.m_frameRange = [int((time_range[0]*self.m_acq0.GetPointFrequency())+self.m_acq0.GetFirstFrame()),int((time_range[1]*self.m_acq0.GetPointFrequency())+self.m_acq0.GetFirstFrame())]

        if not hasattr(self, "m_weights"):
            markertasks = self.xml.m_soup.find_all("IKMarkerTask")
            self.m_weights = dict()
            for item in markertasks:
                self.m_weights[item["name"]] = float(item.find("weight").string)

        ikTool = opensim.InverseKinematicsTool(self.m_ikTool)
        # ikTool.setModel(self.m_osimModel)
        ikTool.run()

        self.finalize()

    def finalize(self):
        # rename the xml setup file with the filename as suffix
        files.renameFile(self.m_ikTool, 
                    self.m_DATA_PATH + self.m_dynamicFile + "-IKTool-setup.xml")
       


class InverseKinematicXmlCgmProcedure(InverseKinematicXmlProcedure):
    def __init__(self,DATA_PATH,scaledOsimName,resultsDirectory,modelVersion):

        super(InverseKinematicXmlCgmProcedure,self).__init__(DATA_PATH,scaledOsimName,resultsDirectory)

        self.m_modelVersion = modelVersion.replace(".", "") if modelVersion is not None else "UnversionedModel"

        if self.m_modelVersion == "CGM23": 
            ikToolFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-ikSetUp_template.xml"

        if self.m_modelVersion == "CGM22": 
            ikToolFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\setup\\CGM22-ikSetUp_template.xml"

        self.m_ikTool = self.m_DATA_PATH + self.m_modelVersion + "-IKTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(ikToolFile,self.m_ikTool)
       
    def finalize(self):
        # rename the xml setup file with the filename as suffix
        files.renameFile(self.m_ikTool, 
                    self.m_DATA_PATH + self.m_dynamicFile+ "-"+self.m_modelVersion + "-IKTool-setup.xml")


class KalmanInverseKinematicXmlCgmProcedure(InverseKinematicXmlProcedure):
    def __init__(self,DATA_PATH,scaledOsimName,resultsDirectory,modelVersion):

        super(KalmanInverseKinematicXmlCgmProcedure,self).__init__(DATA_PATH,scaledOsimName,resultsDirectory)

        self.m_modelVersion = modelVersion.replace(".", "") if modelVersion is not None else "UnversionedModel"

        if self.m_modelVersion == "CGM22": 
            ikToolFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\setup\\CGM22-kalmanIkSetUp_template.xml"
        elif self.m_modelVersion == "CGM23": 
            ikToolFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-kalmanIkSetUp_template.xml"


        self.m_ikTool = self.m_DATA_PATH + self.m_modelVersion + "-kalmanIk-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(ikToolFile,self.m_ikTool)
       
    def prepareWeights(self,weights_dict):
        for key in weights_dict:
            weights_dict[key]=weights_dict[key]/50

        self.m_weights = weights_dict


    def run(self):

        if os.path.isfile(self.m_DATA_PATH +self.m_dynamicFile+"_ik_model_marker_locations.sto"):
            os.remove(self.m_DATA_PATH +self.m_dynamicFile+"_ik_model_marker_locations.sto")
        if os.path.isfile(self.m_DATA_PATH +self.m_dynamicFile+"_ik_marker_errors.sto"):
            os.remove(self.m_DATA_PATH +self.m_dynamicFile+"_ik_marker_errors.sto")

        self.xml.update()

        if not hasattr(self, "m_frameRange"):
            time_range_str = self.xml.m_soup.find("time_range").string
            time_range = [float(it) for it in time_range_str.split(" ")]
            self.m_frameRange = [int((time_range[0]*self.m_acq0.GetPointFrequency())+self.m_acq0.GetFirstFrame()),int((time_range[1]*self.m_acq0.GetPointFrequency())+self.m_acq0.GetFirstFrame())]

        if not hasattr(self, "m_weights"):
            markertasks = self.xml.m_soup.find_all("IKMarkerTask")
            self.m_weights = dict()
            for item in markertasks:
                self.m_weights[item["name"]] = float(item.find("weight").string)

        if self.m_modelVersion == "CGM22":
            cmd = pyCGM2.OPENSIM_KSLIB_PATH+ "ks.exe -S \""+ self.m_DATA_PATH+"CGM22-kalmanIk-setup.xml\""

        if self.m_modelVersion == "CGM23":        
                cmd = pyCGM2.OPENSIM_KSLIB_PATH+ "ks.exe -S \""+ self.m_DATA_PATH+"CGM23-kalmanIk-setup.xml\""
        
        os.system(cmd)
        
        
        # ikTool.setModel(self.m_osimModel)
        

        self.finalize()

    def finalize(self):
        # rename the xml setup file with the filename as suffix
        files.renameFile(self.m_ikTool, 
                    self.m_DATA_PATH + self.m_dynamicFile+ "-"+self.m_modelVersion + "-kalmanIk-setup.xml")

        files.renameFile(self.m_DATA_PATH+self.m_resultsDir + "\\_ks_model_marker_locations.sto", 
                    self.m_DATA_PATH+self.m_resultsDir + "\\_ik_model_marker_locations.sto")

        files.renameFile(self.m_DATA_PATH+self.m_resultsDir + "\\_ks_marker_errors.sto", 
                    self.m_DATA_PATH+self.m_resultsDir + "\\_ik_marker_errors.sto")

