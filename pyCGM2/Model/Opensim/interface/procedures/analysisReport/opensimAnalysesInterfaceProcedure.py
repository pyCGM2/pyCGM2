# -*- coding: utf-8 -*-
import os
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER

# pyCGM2
from pyCGM2.Tools import  btkTools,opensimTools
from pyCGM2.Utils import files
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures
from pyCGM2.Model.Opensim import opensimIO

try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim



class AnalysesXmlProcedure(opensimProcedures.OpensimInterfaceXmlProcedure):

    def __init__(self,DATA_PATH,scaledOsimName,resultsDirectory):
        super(AnalysesXmlProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH
        self.m_osimName = DATA_PATH + scaledOsimName
        self.m_modelVersion = ""
        self.m_resultsDir = "" if resultsDirectory is None else resultsDirectory
        self._externalLoadApplied = True

        files.createDir(self.m_DATA_PATH+self.m_resultsDir)


    def setSetupFiles(self,analysisToolTemplateFile,externalLoadTemplateFile):
        self.m_idAnalyses = self.m_DATA_PATH + "__analysesTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(analysisToolTemplateFile,self.m_idAnalyses)
   
        if externalLoadTemplateFile is not None:
            self.m_externalLoad = self.m_DATA_PATH + "__externalLoad.xml"
            self.xml_load = opensimInterface.opensimXmlInterface(externalLoadTemplateFile,self.m_externalLoad)
        else: 
            self._externalLoadApplied= False

    def setProgression(self,progressionAxis,forwardProgression):
        self.m_progressionAxis = progressionAxis
        self.m_forwardProgression = forwardProgression


    def prepareDynamicTrial(self, acq, dynamicFile,mfpa):
        self.m_dynamicFile =dynamicFile
        self.m_acq = acq
        self.m_mfpa = mfpa

        self.m_ff = self.m_acq.GetFirstFrame()
        self.m_freq = self.m_acq.GetPointFrequency()

        self.m_beginTime = 0
        self.m_endTime = (self.m_acq.GetLastFrame() - self.m_ff)/self.m_freq
        self.m_frameRange = [int((self.m_beginTime*self.m_freq)+self.m_ff),int((self.m_endTime*self.m_freq)+self.m_ff)] 

        if self._externalLoadApplied:
            opensimTools.footReactionMotFile(
                self.m_acq, self.m_DATA_PATH+self.m_resultsDir+"\\"+self.m_dynamicFile+"_grf.mot",
                self.m_progressionAxis,self.m_forwardProgression,mfpa = self.m_mfpa)
        else:
            LOGGER.logger.info("Trial without force plate - No dynamic trials prepared ")


    def setTimeRange(self,beginFrame=None,lastFrame=None):

        self.m_beginTime = 0.0 if beginFrame is None else (beginFrame-self.m_ff)/self.m_freq
        self.m_endTime = (self.m_acq.GetLastFrame() - self.m_ff)/self.m_freq  if lastFrame is  None else (lastFrame-self.m_ff)/self.m_freq

    def prepareXml(self):


        # self.xml.set_one("model_file", self.m_dynamicFile+".mot")
        self.xml.getSoup().find("AnalyzeTool").attrs["name"] = self.m_dynamicFile+"-analyses"

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_DATA_PATH+self.m_resultsDir+ "\\"+self.m_dynamicFile+".mot")
        self.xml.set_one("results_directory",  self.m_resultsDir)
        self.xml.set_one("initial_time",str(self.m_beginTime))
        self.xml.set_one("final_time",str(self.m_endTime))

        if self._externalLoadApplied:
            self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))
           
            self.xml_load.set_one("datafile", self.m_DATA_PATH+self.m_resultsDir + "\\"+ self.m_dynamicFile+"_grf.mot")


    def run(self):

        self.xml.update()
        if self._externalLoadApplied:
            self.xml_load.update()


        tool = opensim.AnalyzeTool(self.m_idAnalyses)
        tool.run()

        self.finalize()

    def finalize(self):
        files.renameFile(self.m_idAnalyses, 
            self.m_DATA_PATH + self.m_dynamicFile+ "-analysesTool-setup.xml")

        if self._externalLoadApplied:
            files.renameFile(self.m_externalLoad,
                self.m_DATA_PATH + self.m_dynamicFile + "-externalLoad.xml")


class AnalysesXmlCgmProcedure(AnalysesXmlProcedure):

    def __init__(self,DATA_PATH,scaledOsimName, resultsDirectory,modelVersion):
        super(AnalysesXmlCgmProcedure,self).__init__(DATA_PATH,scaledOsimName,resultsDirectory)

        self.m_modelVersion = modelVersion.replace(".", "") if modelVersion is not None else "UnversionedModel"

        if self.m_modelVersion == "CGM23":
            analysisToolTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-muscleAnalysisSetup_template.xml"
            externalLoadTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\walk_grf.xml"

        if self.m_modelVersion == "CGM22":
            analysisToolTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\setup\\CGM22-muscleAnalysisSetup_template.xml"
            externalLoadTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\setup\\walk_grf.xml"


        self.m_idAnalyses = self.m_DATA_PATH + self.m_modelVersion + "-analysesTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(analysisToolTemplateFile,self.m_idAnalyses)
   
        if externalLoadTemplateFile is not None:
            self.m_externalLoad = self.m_DATA_PATH + self.m_modelVersion + "-externalLoad.xml"
            self.xml_load = opensimInterface.opensimXmlInterface(externalLoadTemplateFile,self.m_externalLoad)
        else: 
            self._externalLoadApplied= False

    def prepareXml(self):


        # self.xml.set_one("model_file", self.m_dynamicFile+".mot")
        self.xml.getSoup().find("AnalyzeTool").attrs["name"] = self.m_dynamicFile+"-"+self.m_modelVersion+"-analyses"

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_DATA_PATH+self.m_resultsDir+ "\\"+self.m_dynamicFile+".mot")
        self.xml.set_one("results_directory",  self.m_resultsDir)
        self.xml.set_one("initial_time",str(self.m_beginTime))
        self.xml.set_one("final_time",str(self.m_endTime))

        if self._externalLoadApplied:
            self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))
           
            self.xml_load.set_one("datafile", self.m_DATA_PATH+self.m_resultsDir + "\\"+ self.m_dynamicFile+"_grf.mot")

    def finalize(self):
        files.renameFile(self.m_idAnalyses, 
            self.m_DATA_PATH + self.m_dynamicFile+ "-"+self.m_modelVersion + "-analysesTool-setup.xml")

        if self._externalLoadApplied:
            files.renameFile(self.m_externalLoad,
                self.m_DATA_PATH + self.m_dynamicFile + "-"+self.m_modelVersion + "-externalLoad.xml")


class AnalysesXmlCgmDrivenModelProcedure(AnalysesXmlCgmProcedure):

    def __init__(self,DATA_PATH,scaledOsimName,resultsDirectory,modelVersion):
        super(AnalysesXmlCgmDrivenModelProcedure,self).__init__(DATA_PATH,scaledOsimName,resultsDirectory,modelVersion)

        self.m_acq = None
    
        if self.m_modelVersion == "CGM23":
            self.m_refPose = opensimIO.OpensimDataFrame(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\", "referencePose.mot")
        
        elif self.m_modelVersion == "CGM22":
            self.m_refPose = opensimIO.OpensimDataFrame(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\", "referencePose.mot")

        self.m_beginTime = 0
        self.m_endTime = self.m_refPose.getDataFrame()["time"].iloc[-1]

        self.m_poseLabel = ""

        self._externalLoadApplied = False
        

    def setPose(self,poseLabel,qi = None):
        self.m_poseLabel = poseLabel

        if qi is not None:
            for key in qi:
                self.m_refPose.getDataFrame()[key] = qi[key]
                # self.m_refPose.getDataFrame()["hip_flexion_r"] = 90.0#np.deg2rad(90.0)
                # self.m_refPose.getDataFrame()["knee_flexion_r"] = -90.0#np.deg2rad(-90.0)
        self.m_refPose.save( outDir = self.m_DATA_PATH+self.m_resultsDir+"\\" , filename=poseLabel+".mot")
        
        self.m_poseLabel+".mot"

    def forceMotFile(self,motfile):
        self.m_poseLabel = motfile[:-4]

        files.copyPaste(self.m_DATA_PATH+ motfile,self.m_DATA_PATH+self.m_resultsDir+"\\"+motfile)
        self.m_refPose = opensimIO.OpensimDataFrame(self.m_DATA_PATH+self.m_resultsDir+"\\", motfile)

        self.m_beginTime = 0
        self.m_endTime = self.m_refPose.getDataFrame()["time"].iloc[-1]
    

    
    def prepareXml(self):


        # self.xml.set_one("model_file", self.m_dynamicFile+".mot")
        self.xml.getSoup().find("AnalyzeTool").attrs["name"] = self.m_modelVersion +"-Pose["+self.m_poseLabel+"]"

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_DATA_PATH+self.m_resultsDir+"\\"+self.m_poseLabel+".mot")
        self.xml.set_one("results_directory",  self.m_resultsDir)
        self.xml.set_one("initial_time",str(self.m_beginTime))
        self.xml.set_one("final_time",str(self.m_endTime))

        if self._externalLoadApplied:
            self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))
           
            self.xml_load.set_one("datafile", self.m_DATA_PATH+self.m_resultsDir + "\\"+ self.m_dynamicFile+"_grf.mot")


    def finalize(self):
        files.renameFile(self.m_idAnalyses, 
            self.m_DATA_PATH + self.m_modelVersion + "-Pose["+self.m_poseLabel+"]-analysesTool-setup.xml")

        if self._externalLoadApplied:
            files.renameFile(self.m_externalLoad,
                self.m_DATA_PATH + self.m_modelVersion + "-Pose["+self.m_poseLabel+"]-externalLoad.xml")

