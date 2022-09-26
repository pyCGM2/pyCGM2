# -*- coding: utf-8 -*-
import os
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from bs4 import BeautifulSoup

# pyCGM2
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk
from pyCGM2.Tools import  btkTools,opensimTools
from pyCGM2.Model.Opensim.interface import opensimInterfaceFilters
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures
from pyCGM2.Utils import files
try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim
from pyCGM2.Model.Opensim import opensimIO


class InverseKinematicXMLProcedure(opensimProcedures.OpensimInterfaceXmlProcedure):
    def __init__(self,DATA_PATH, scaledOsimName,modelVersion,ikToolTemplateFile,
                local=False):

        super(InverseKinematicXMLProcedure,self).__init__()
        self.m_DATA_PATH = DATA_PATH
        self.m_resultsDir = ""

        # self.m_osimModel = scaleOsim
        self.m_osimName = DATA_PATH + scaledOsimName
        self.m_modelVersion = modelVersion.replace(".", "")

        if not local:
            if ikToolTemplateFile is None:
                raise Exception("ikToolTemplateFile needs to be defined")
            self.m_ikTool = DATA_PATH + self.m_modelVersion + "-IKTool-setup.xml"
            self.xml = opensimInterfaceFilters.opensimXmlInterface(ikToolTemplateFile,self.m_ikTool)
        else:
            self.m_ikTool = DATA_PATH + ikToolTemplateFile
            self.xml = opensimInterfaceFilters.opensimXmlInterface(self.m_ikTool)

        self.m_accuracy = 1e-8
    
    def setProgression(self,progressionAxis,forwardProgression):
        self.m_progressionAxis = progressionAxis
        self.m_forwardProgression = forwardProgression
        self.m_R_LAB_OSIM = opensimTools.rotationMatrix_labToOsim(progressionAxis,forwardProgression)


    def prepareDynamicTrial(self, acq, dynamicFile):

        self.m_dynamicFile = dynamicFile
        self.m_acq0 = acq
        self.m_acqMotion_forIK = btk.btkAcquisition.Clone(acq)

        self.m_ff = self.m_acqMotion_forIK.GetFirstFrame()
        self.m_freq = self.m_acqMotion_forIK.GetPointFrequency()

        opensimTools.transformMarker_ToOsimReferencial(self.m_acqMotion_forIK,self.m_progressionAxis,self.m_forwardProgression)

        self.m_markerFile = btkTools.smartWriter(self.m_acqMotion_forIK,self.m_DATA_PATH +  dynamicFile, extension="trc")

        self.m_beginTime = 0
        self.m_endTime = (self.m_acqMotion_forIK.GetLastFrame() - self.m_ff)/self.m_freq
        self.m_frameRange = [int((self.m_beginTime*self.m_freq)+self.m_ff),int((self.m_endTime*self.m_freq)+self.m_ff)] 

    def setWeights(self,weights_dict):
        self.m_weights = weights_dict

    def setAccuracy(self,value):
        self.m_accuracy = value

    def setTimeRange(self,beginFrame=None,lastFrame=None):

        self.m_beginTime = 0.0 if beginFrame is None else (beginFrame-self.m_ff)/self.m_freq
        self.m_endTime = (self.m_acqMotion_forIK.GetLastFrame() - self.m_ff)/self.m_freq  if lastFrame is  None else (lastFrame-self.m_ff)/self.m_freq
        
        self.m_frameRange = [int((self.m_beginTime*self.m_freq)+self.m_ff),int((self.m_endTime*self.m_freq)+self.m_ff)]

    def setResultsDirname(self,dirname):
        self.m_resultsDir = dirname

    def _prepareXml(self):

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("marker_file", files.getFilename(self.m_markerFile))
        self.xml.set_one("output_motion_file", self.m_dynamicFile+".mot")
        for marker in self.m_weights.keys():
            self.xml.set_inList_fromAttr("IKMarkerTask","weight","name",marker,str(self.m_weights[marker]))

        self.xml.set_one("accuracy",str(self.m_accuracy))
        self.xml.set_one("time_range",str(self.m_beginTime) + " " + str(self.m_endTime))

        if self.m_resultsDir !="":
            self.xml.set_one("results_directory",  self.m_resultsDir)

    def run(self):

        if os.path.isfile(self.m_DATA_PATH +self.m_dynamicFile+"_ik_model_marker_locations.sto"):
            os.remove(self.m_DATA_PATH +self.m_dynamicFile+"_ik_model_marker_locations.sto")
        if os.path.isfile(self.m_DATA_PATH +self.m_dynamicFile+"_ik_marker_errors.sto"):
            os.remove(self.m_DATA_PATH +self.m_dynamicFile+"_ik_marker_errors.sto")

        if self.m_autoXml: self._prepareXml()
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
        pass
