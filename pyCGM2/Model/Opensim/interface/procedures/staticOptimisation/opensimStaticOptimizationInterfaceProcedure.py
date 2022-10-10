# -*- coding: utf-8 -*-
import os
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER


# pyCGM2
from pyCGM2.Tools import  opensimTools
from pyCGM2.Utils import files
from pyCGM2.Model.Opensim.interface import opensimInterfaceFilters
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures

try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim



class StaticOptimisationXmlProcedure(opensimProcedures.OpensimInterfaceXmlProcedure):
    def __init__(self,DATA_PATH,scaledOsimName,resultsDirectory):
        super(StaticOptimisationXmlProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH
        self.m_osimName = DATA_PATH + scaledOsimName
        self.m_resultsDir = "" if resultsDirectory is None else resultsDirectory

        files.createDir(self.m_DATA_PATH+self.m_resultsDir)
        self.m_modelVersion = ""

    def setSetupFiles(self,analysisToolTemplateFile,externalLoadTemplateFile):
        self.m_soTool = self.m_DATA_PATH + "__SOTool-setup.xml"
        self.xml = opensimInterfaceFilters.opensimXmlInterface(analysisToolTemplateFile,self.m_soTool)
   
        self.m_externalLoad = self.m_DATA_PATH + "__externalLoad.xml"
        self.xml_load = opensimInterfaceFilters.opensimXmlInterface(externalLoadTemplateFile,self.m_externalLoad)
    
    def setProgression(self,progressionAxis,forwardProgression):
        self.m_progressionAxis = progressionAxis
        self.m_forwardProgression = forwardProgression


    def setResultsDirname(self,dirname):
        self.m_resultsDir = dirname    

    def prepareDynamicTrial(self, acq, dynamicFile,mfpa):
        self.m_dynamicFile =dynamicFile
        self.m_acq = acq
        self.m_mfpa = mfpa

        self.m_ff = self.m_acq.GetFirstFrame()
        self.m_freq = self.m_acq.GetPointFrequency()

        self.m_beginTime = 0
        self.m_endTime = (self.m_acq.GetLastFrame() - self.m_ff)/self.m_freq
        self.m_frameRange = [int((self.m_beginTime*self.m_freq)+self.m_ff),int((self.m_endTime*self.m_freq)+self.m_ff)] 

        opensimTools.footReactionMotFile(
            self.m_acq, self.m_DATA_PATH+self.m_resultsDir+"\\"+self.m_dynamicFile+"_grf.mot",
            self.m_progressionAxis,self.m_forwardProgression,mfpa = self.m_mfpa)


    def setTimeRange(self,beginFrame=None,lastFrame=None):

        self.m_beginTime = 0.0 if beginFrame is None else (beginFrame-self.m_ff)/self.m_freq
        self.m_endTime = (self.m_acq.GetLastFrame() - self.m_ff)/self.m_freq  if lastFrame is  None else (lastFrame-self.m_ff)/self.m_freq



    def prepareXml(self):

        # self.xml.set_one("model_file", self.m_dynamicFile+".mot")
        self.xml.getSoup().find("AnalyzeTool").attrs["name"] = self.m_dynamicFile+"-analyses"

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_DATA_PATH+self.m_resultsDir + "\\"+ self.m_dynamicFile+".mot")
        self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))

        self.xml.set_one("results_directory",  self.m_resultsDir)

        self.xml.set_one("initial_time",str(self.m_beginTime))
        self.xml.set_one("final_time",str(self.m_endTime))

        self.xml.m_soup.AnalysisSet.start_time.string =  str(self.m_beginTime)
        self.xml.m_soup.AnalysisSet.end_time.string =  str(self.m_endTime)

        self.xml_load.set_one("datafile", self.m_DATA_PATH+self.m_resultsDir + "\\"+ self.m_dynamicFile+"_grf.mot")


    def run(self):

        self.xml.update()
        self.xml_load.update()

        tool = opensim.AnalyzeTool(self.m_soTool)
        tool.run()

        self.finalize()

    def finalize(self):


        files.renameFile(self.m_soTool, 
            self.m_DATA_PATH + self.m_dynamicFile+ "-SOTool-setup.xml")
        files.renameFile(self.m_externalLoad,
             self.m_DATA_PATH + self.m_dynamicFile + "-externalLoad.xml")


class StaticOptimisationXmlCgmProcedure(StaticOptimisationXmlProcedure):
    def __init__(self,DATA_PATH,scaledOsimName, modelVersion,resultsDirectory):
        super(StaticOptimisationXmlCgmProcedure,self).__init__(DATA_PATH,scaledOsimName,resultsDirectory)

        self.m_modelVersion = modelVersion.replace(".", "") if modelVersion is not None else "UnversionedModel"

        if self.m_modelVersion == "CGM23":
            analysisToolTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\setup\\CGM23\\CGM23-soSetup_template.xml"
            externalLoadTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\setup\\walk_grf.xml"

        self.m_soTool = self.m_DATA_PATH + self.m_modelVersion + "-SOTool-setup.xml"
        self.xml = opensimInterfaceFilters.opensimXmlInterface(analysisToolTemplateFile,self.m_soTool)
   
        self.m_externalLoad = self.m_DATA_PATH + self.m_modelVersion + "-externalLoad.xml"
        self.xml_load = opensimInterfaceFilters.opensimXmlInterface(externalLoadTemplateFile,self.m_externalLoad)
    
   
    def prepareXml(self):

        # self.xml.set_one("model_file", self.m_dynamicFile+".mot")
        self.xml.getSoup().find("AnalyzeTool").attrs["name"] = self.m_dynamicFile+"-"+self.m_modelVersion+"-analyses"

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_DATA_PATH+self.m_resultsDir + "\\"+ self.m_dynamicFile+".mot")
        self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))

        self.xml.set_one("results_directory",  self.m_resultsDir)

        self.xml.set_one("initial_time",str(self.m_beginTime))
        self.xml.set_one("final_time",str(self.m_endTime))

        self.xml.m_soup.AnalysisSet.start_time.string =  str(self.m_beginTime)
        self.xml.m_soup.AnalysisSet.end_time.string =  str(self.m_endTime)

        self.xml_load.set_one("datafile", self.m_DATA_PATH+self.m_resultsDir + "\\"+ self.m_dynamicFile+"_grf.mot")


    def finalize(self):


        files.renameFile(self.m_soTool, 
            self.m_DATA_PATH + self.m_dynamicFile+ "-"+self.m_modelVersion + "-SOTool-setup.xml")
        files.renameFile(self.m_externalLoad,
             self.m_DATA_PATH + self.m_dynamicFile + "-"+self.m_modelVersion + "-externalLoad.xml")