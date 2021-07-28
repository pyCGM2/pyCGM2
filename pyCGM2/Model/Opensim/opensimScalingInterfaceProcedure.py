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
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.Opensim import osimProcessing
from pyCGM2.Processing import progressionFrame

from pyCGM2.Model.Opensim import opensimInterfaceFilters

from pyCGM2.Utils import files
try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim



class opensimInterfaceHighLevelScalingProcedure(object):
    def __init__(self,DATA_PATH, osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile,statictrcFile):

        self.m_DATA_PATH = DATA_PATH
        self.m_staticFile = statictrcFile


        self.m_osimTemplateFullFile = osimTemplateFullFile
        self.m_short_osimTemplateFullFile = self.m_osimTemplateFullFile[len(os.path.dirname(self.m_osimTemplateFullFile))+1:]
        files.copyPaste(osimTemplateFullFile,DATA_PATH+self.m_short_osimTemplateFullFile)

        self.m_markersetTemplateFullFile = markersetTemplateFullFile
        self.m_short_markersetTemplateFullFile = self.m_markersetTemplateFullFile[len(os.path.dirname(self.m_markersetTemplateFullFile))+1:]
        files.copyPaste(markersetTemplateFullFile,DATA_PATH+self.m_short_markersetTemplateFullFile)


        self.m_scaleToolFile = DATA_PATH + "ScaleTool-setup.xml" #"/scaleTool.xml"
        self.xml = opensimInterfaceFilters.opensimXmlInterface(scaleToolFullFile,self.m_scaleToolFile)

    def setAnthropometry(self, mass, height):
        self.xml.set_one("mass", str(mass))
        self.xml.set_one("height", str(height))

    def _timeRangeFromStatic(self):
        static = opensim.MarkerData(self.m_DATA_PATH+ self.m_staticFile)
        initial_time = static.getStartFrameTime()
        final_time = static.getLastFrameTime()

        text = str(initial_time) + " " + str(final_time)
        self.xml.set_many("time_range",text)

    def run(self):

        self.xml.getSoup().GenericModelMaker.model_file.string =    self.m_short_osimTemplateFullFile
        self.xml.getSoup().GenericModelMaker.marker_set_file.string =    self.m_short_markersetTemplateFullFile
        self._timeRangeFromStatic()

        self.xml.set_many("marker_file",self.m_staticFile)

        self.xml.update()

        scale_tool = opensim.ScaleTool(self.m_scaleToolFile)
        scale_tool.run()

        self.m_osimModel = opensim.Model( self.m_DATA_PATH+"__ScaledModel.osim")



class opensimInterfaceLowLevelScalingProcedure(object):
    def __init__(self,DATA_PATH, osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile,statictrcFile):

        self.m_DATA_PATH = DATA_PATH


        self.m_staticFile = self.m_DATA_PATH+ statictrcFile

        # initialize scale tool from setup file
        self.m_osimModel = opensim.Model(osimTemplateFullFile)
        self.m_osimModel.setName("pyCGM2-Opensim-scaled")
        markerSet= opensim.MarkerSet(markersetTemplateFullFile)
        self.m_osimModel.updateMarkerSet(markerSet)

        # initialize scale tool from setup file
        self.m_scale_tool = opensim.ScaleTool(scaleToolFullFile)
        self.m_scale_tool.getGenericModelMaker().setModelFileName(osimTemplateFullFile)


        self._MODEL_OUTPUT = self.m_DATA_PATH+"__ScaledCGM.osim"
        self._MODEL_MARKERS_OUTPUT = self._MODEL_OUTPUT.replace(".osim", "_markers.osim")
        self._SCALETOOL_OUTPUT = self.m_DATA_PATH+"__scaleTool.xml"




    def _timeRangeFromStatic(self):
        static = opensim.MarkerData(self.m_staticFile)
        initial_time = static.getStartFrameTime()
        final_time = static.getLastFrameTime()
        range_time = opensim.ArrayDouble()
        range_time.set(0, initial_time)
        range_time.set(1, final_time)
        return range_time

    def setAnthropometry(self, mass, height):#, age):
        self.m_scale_tool.setSubjectMass(mass)
        self.m_scale_tool.setSubjectHeight(height)


    def run(self):

        time_range = self._timeRangeFromStatic()

        #---model_scaler---
        model_scaler = self.m_scale_tool.getModelScaler()
        model_scaler.setApply(True)
        # Set the marker file to be used for scaling
        model_scaler.setMarkerFileName(self.m_staticFile)
        # set time range
        model_scaler.setTimeRange(time_range)
        # Indicating whether or not to preserve relative mass between segments
        model_scaler.setPreserveMassDist(True)

        # Name of model file (.osim) to write when done scaling
        model_scaler.setOutputModelFileName(self._MODEL_OUTPUT)

        # # Filename to write scale factors that were applied to the unscaled model (optional)
        # model_scaler.setOutputScaleFileName(
        #     self.scaleTool_output.replace(".xml", "_scaling_factor.xml")
        # )

        model_scaler.processModel(self.m_osimModel)


        #---marker_scaler---
        # self.m_osimModel = opensim.Model(self.__MODEL_OUTPUT)

        marker_placer = self.m_scale_tool.getMarkerPlacer()
        # Whether or not to use the model scaler during scale`
        marker_placer.setApply(False)
        marker_placer.setTimeRange(time_range)

        marker_placer.setStaticPoseFileName(self.m_staticFile)

        # Name of model file (.osim) to write when done scaling
        marker_placer.setOutputModelFileName(self._MODEL_MARKERS_OUTPUT)

        # Maximum amount of movement allowed in marker data when averaging
        marker_placer.setMaxMarkerMovement(-1)

        marker_placer.processModel(self.m_osimModel)

        self.m_scale_tool.printToXML(self._SCALETOOL_OUTPUT)
