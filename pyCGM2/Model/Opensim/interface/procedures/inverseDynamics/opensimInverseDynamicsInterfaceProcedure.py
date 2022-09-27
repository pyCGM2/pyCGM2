# -*- coding: utf-8 -*-
import os
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# pyCGM2
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk
from pyCGM2.Tools import  btkTools,opensimTools
from pyCGM2.Utils import files
from pyCGM2.Model.Opensim.interface import opensimInterfaceFilters
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures

try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim



class InverseDynamicsXMLProcedure(opensimProcedures.OpensimInterfaceXmlProcedure):
    def __init__(self,DATA_PATH, scaledOsimName,modelVersion,idToolTemplateFile,externalLoadTemplateFile,
        mfpa = None,
        local=False):

        super(InverseDynamicsXMLProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH
        self.m_resultsDir = ""

        self.m_osimName = DATA_PATH + scaledOsimName
        self.m_modelVersion = modelVersion.replace(".", "")

        self.m_mfpa = mfpa

        if not local:
            if idToolTemplateFile is None:
                raise Exception("idToolTemplateFile needs to be defined")

            self.m_idTool = DATA_PATH + self.m_modelVersion + "-idTool-setup.xml"
            self.xml = opensimInterfaceFilters.opensimXmlInterface(idToolTemplateFile,self.m_idTool)
        else:
            self.m_idTool = DATA_PATH + idToolTemplateFile
            self.xml = opensimInterfaceFilters.opensimXmlInterface(self.m_idTool,None)

        if not local:
            if externalLoadTemplateFile is None:
                raise Exception("externalLoadTemplateFile needs to be defined")

            self.m_externalLoad = DATA_PATH + self.m_modelVersion + "-externalLoad.xml"
            self.xml_load = opensimInterfaceFilters.opensimXmlInterface(externalLoadTemplateFile,self.m_externalLoad)

        else:
            self.m_externalLoad = DATA_PATH + externalLoadTemplateFile
            self.xml_load = opensimInterfaceFilters.opensimXmlInterface(self.m_externalLoad,None)


    def setProgression(self,progressionAxis,forwardProgression):
        self.m_progressionAxis = progressionAxis
        self.m_forwardProgression = forwardProgression



    def prepareDynamicTrial(self, acq, dynamicFile):
        self.m_dynamicFile =dynamicFile
        self.m_acq = acq

        self.m_ff = self.m_acq.GetFirstFrame()
        self.m_freq = self.m_acq.GetPointFrequency()

        self.m_beginTime = 0
        self.m_endTime = (self.m_acq.GetLastFrame() - self.m_ff)/self.m_freq
        self.m_frameRange = [int((self.m_beginTime*self.m_freq)+self.m_ff),int((self.m_endTime*self.m_freq)+self.m_ff)] 

        opensimTools.footReactionMotFile(
            self.m_acq, self.m_DATA_PATH+self.m_dynamicFile+"_grf.mot",
            self.m_progressionAxis,self.m_forwardProgression,mfpa = self.m_mfpa)

    def setResultsDirname(self,dirname):
        self.m_resultsDir = dirname

    def setTimeRange(self,beginFrame=None,lastFrame=None):

        self.m_beginTime = 0.0 if beginFrame is None else (beginFrame-self.m_ff)/self.m_freq
        self.m_endTime = (self.m_acq.GetLastFrame() - self.m_ff)/self.m_freq  if lastFrame is  None else (lastFrame-self.m_ff)/self.m_freq
        
        self.m_frameRange = [int((self.m_beginTime*self.m_freq)+self.m_ff),int((self.m_endTime*self.m_freq)+self.m_ff)]


    def _prepareXml(self):
        self.xml.getSoup().find("InverseDynamicsTool").attrs["name"] = self.m_modelVersion+"-InverseDynamics"
        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_dynamicFile+".mot")
        self.xml.set_one("output_gen_force_file", self.m_dynamicFile+"-"+self.m_modelVersion+"-inverse_dynamics.sto")
        self.xml.set_one("lowpass_cutoff_frequency_for_coordinates","6")

        self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))

        self.xml.set_one("time_range",str(self.m_beginTime) + " " + str(self.m_endTime))

        if self.m_resultsDir !="":
            self.xml.set_one("results_directory",  self.m_resultsDir)


        self.xml_load.set_one("datafile", self.m_dynamicFile+"_grf.mot")
        

    def run(self):

        if self.m_autoXml:
            self._prepareXml()

        self.xml.update()
        self.xml_load.update()


        idTool = opensim.InverseDynamicsTool(self.m_idTool)
        # idTool.setModel(self.m_osimModel)
        idTool.run()

        self.finalize()
    
    def finalize(self):

        files.renameFile(self.m_idTool, 
                    self.m_DATA_PATH + self.m_dynamicFile+ "-"+self.m_modelVersion + "-IDTool-setup.xml")
        files.renameFile(self.m_externalLoad,
             self.m_DATA_PATH + self.m_dynamicFile + self.m_modelVersion + "-externalLoad.xml")


