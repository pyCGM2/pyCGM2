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
from pyCGM2.Model.Opensim import osimProcessing
from pyCGM2.Processing import progressionFrame
from pyCGM2.Utils import files
from pyCGM2.Model.Opensim import opensimInterfaceFilters
from pyCGM2.Model.Opensim import opensimIO
try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim



class highLevelInverseDynamicsProcedure(object):
    def __init__(self,DATA_PATH, scaledOsimName,modelVersion,idToolTemplateFile,externalLoadTemplateFile,
        mfpa = None,
        localIdToolFile=None,
        localExternalLoadFile=None):

        self.m_DATA_PATH = DATA_PATH
        self._resultsDir = ""

        self.m_osimName = scaledOsimName
        self.m_modelVersion = modelVersion.replace(".", "")

        self.m_mfpa = mfpa

        if localIdToolFile is None:
            if idToolTemplateFile is None:
                raise Exception("localIdToolFile or idToolTemplateFile needs to be defined")

            self.m_idTool = DATA_PATH + self.m_modelVersion + "-idTool-setup.xml"
            self.xml = opensimInterfaceFilters.opensimXmlInterface(idToolTemplateFile,self.m_idTool)
        else:
            self.m_idTool = DATA_PATH + localIdToolFile
            self.xml = opensimInterfaceFilters.opensimXmlInterface(self.m_idTool,None)

        if localExternalLoadFile is None:
            if externalLoadTemplateFile is None:
                raise Exception("localExternalLoadFile or externalLoadTemplateFile needs to be defined")

            self.m_externalLoad = DATA_PATH + self.m_modelVersion + "-externalLoad.xml"
            self.xml_load = opensimInterfaceFilters.opensimXmlInterface(externalLoadTemplateFile,self.m_externalLoad)

        else:
            self.m_externalLoad = DATA_PATH+localExternalLoadFile
            self.xml_load = opensimInterfaceFilters.opensimXmlInterface(self.m_externalLoad,None)

        self.m_autoXmlDefinition=True

    def setProgression(self,progressionAxis,forwardProgression):
        self.m_progressionAxis = progressionAxis
        self.m_forwardProgression = forwardProgression

    def setAutoXmlDefinition(self,boolean):
        self.m_autoXmlDefinition=boolean

    def preProcess(self, acq, dynamicFile):
        self.m_dynamicFile =dynamicFile
        self.m_acq = acq

        opensimTools.footReactionMotFile(
            self.m_acq, self.m_DATA_PATH+self.m_dynamicFile+"_grf.mot",
            self.m_progressionAxis,self.m_forwardProgression,mfpa = self.m_mfpa)

    def setResultsDirname(self,dirname):
        self.xml.set_one("results_directory", dirname)
        self._resultsDir = dirname

    def setTimeRange(self,beginFrame=None,lastFrame=None):

        ff = self.m_acq.GetFirstFrame()
        freq = self.m_acq.GetPointFrequency()
        beginTime = 0.0 if beginFrame is None else (beginFrame-ff)/freq
        endTime = (self.m_acq.GetLastFrame() - ff)/freq  if lastFrame is  None else (lastFrame-ff)/freq
        text = str(beginTime) + " " + str(endTime)
        self.xml.set_one("time_range",text)

        self.m_frameRange = [int((beginTime*freq)+ff),int((endTime*freq)+ff)]

    def _setXml(self):
        self.xml.getSoup().find("InverseDynamicsTool").attrs["name"] = self.m_modelVersion+"-InverseDynamics"
        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_dynamicFile+".mot")
        # self.xml.set_one("results_directory", self.m_DATA_PATH)
        self.xml.set_one("output_gen_force_file", self.m_dynamicFile+"-"+self.m_modelVersion+"-inverse_dynamics.sto")
        self.xml.set_one("lowpass_cutoff_frequency_for_coordinates","6")

        self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))

    def _setXmlLoad(self):

        self.xml_load.set_one("datafile", self.m_dynamicFile+"_grf.mot")

    def run(self):

        if self.m_autoXmlDefinition:
            self._setXml()
            self._setXmlLoad()

        self.xml.update()
        self.xml_load.update()


        idTool = opensim.InverseDynamicsTool(self.m_idTool)
        # idTool.setModel(self.m_osimModel)
        idTool.run()

        self.finalize()
    def finalize(self):
        pass


# NOT WORK : need opensim4.2 and bug fix of property
class opensimInterfaceLowLevelInversedynamicsProcedure(object):
    def __init__(self,DATA_PATH, scaleOsim,idToolsTemplate):
        pass
