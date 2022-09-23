# -*- coding: utf-8 -*-
from distutils import extension
from pyCGM2.Utils import files
from pyCGM2.Tools import btkTools
from pyCGM2.Model.Opensim.interface import opensimInterfaceFilters
import pyCGM2
LOGGER = pyCGM2.LOGGER

# pyCGM2
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk

try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim


class highLevelScalingProcedure(object):
    def __init__(self, DATA_PATH, modelVersion, osimTemplateFile, markersetTemplateFile, scaleToolTemplateFile,
                local=False):

        self.m_DATA_PATH = DATA_PATH
        self.m_modelVersion = modelVersion.replace(".", "")

        if not local :
            if osimTemplateFile is None:
                raise Exception("osimTemplateFile needs to be defined")
            self.m_osim = files.getFilename(osimTemplateFile)
            files.copyPaste(osimTemplateFile, DATA_PATH + self.m_osim)
        else:
            self.m_osim = self.m_DATA_PATH+osimTemplateFile

        if not local:
            if markersetTemplateFile is None:
                raise Exception("localMarkersetFile or markersetTemplateFile needs to be defined")
            self.m_markerset =  files.getFilename(markersetTemplateFile)
            files.copyPaste(markersetTemplateFile, DATA_PATH + self.m_markerset)
        else:
            self.m_markerset = self.m_DATA_PATH+markersetTemplateFile

        if not local:
            if scaleToolTemplateFile is None:
                raise Exception("scaleToolTemplateFile needs to be defined")
            self.m_scaleTool = DATA_PATH + self.m_modelVersion+"-ScaleToolSetup.xml" #files.getFilename(scaleToolTemplateFile)
            self.xml = opensimInterfaceFilters.opensimXmlInterface(scaleToolTemplateFile, self.m_scaleTool)
        else:
            self.m_scaleTool = DATA_PATH + scaleToolTemplateFile
            self.xml = opensimInterfaceFilters.opensimXmlInterface(DATA_PATH+scaleToolTemplateFile, None)

        self.m_autoXmlDefinition=True

    def setAutoXmlDefinition(self,boolean):
        self.m_autoXmlDefinition=boolean

    def preProcess(self, acq, staticFileNoExt):
        self._staticTrc = btkTools.smartWriter( acq, self.m_DATA_PATH + staticFileNoExt, extension="trc")

    def setAnthropometry(self, mass, height):
        self.xml.set_one("mass", str(mass))
        self.xml.set_one("height", str(height))

    def _timeRangeFromStatic(self):
        static = opensim.MarkerData(self._staticTrc)
        initial_time = static.getStartFrameTime()
        final_time = static.getLastFrameTime()

        text = str(initial_time) + " " + str(final_time)
        self.xml.set_many("time_range", text)

    def getXml(self):
        return self.xml

    def _setXml(self):
        # self.xml.getSoup().GenericModelMaker.model_file.string = self.m_osim
        # self.xml.getSoup().GenericModelMaker.marker_set_file.string = self.m_markerset
        # self.xml.getSoup().MarkerPlacer.output_model_file.string =  self.m_modelVersion+"-ScaledModel.osim"
        self.xml.getSoup().find("ScaleTool").attrs["name"] = self.m_modelVersion+"-Scale"
        self.xml.set_one(["GenericModelMaker","model_file"],self.m_osim)
        self.xml.set_one(["GenericModelMaker","marker_set_file"],self.m_markerset)
        self._timeRangeFromStatic()
        self.xml.set_many("marker_file", files.getFilename(self._staticTrc))
        self.xml.set_one(["MarkerPlacer","output_model_file"],self.m_modelVersion+"-ScaledModel.osim")


    def run(self):

        if self.m_autoXmlDefinition: self._setXml()
        self.xml.update()

        scale_tool = opensim.ScaleTool(self.m_scaleTool)
        
        scale_tool.run()
        self.m_osimModel_name = self.m_DATA_PATH+self.m_modelVersion+"-ScaledModel.osim"
        self.m_osimModel = opensim.Model(self.m_DATA_PATH+self.m_modelVersion+"-ScaledModel.osim")


class opensimInterfaceLowLevelScalingProcedure(object):
    def __init__(self, DATA_PATH, osimTemplateFullFile, markersetTemplateFullFile, scaleToolFullFile):

        self.m_DATA_PATH = DATA_PATH

        # initialize scale tool from setup file
        self.m_osimModel = opensim.Model(osimTemplateFullFile)
        self.m_osimModel.setName("pyCGM2-Opensim-scaled")
        markerSet = opensim.MarkerSet(markersetTemplateFullFile)
        self.m_osimModel.updateMarkerSet(markerSet)

        # initialize scale tool from setup file
        self.m_scale_tool = opensim.ScaleTool(scaleToolFullFile)
        self.m_scale_tool.getGenericModelMaker().setModelFileName(osimTemplateFullFile)

        self._MODEL_OUTPUT = self.m_DATA_PATH+"__ScaledCGM.osim"
        self._MODEL_MARKERS_OUTPUT = self._MODEL_OUTPUT.replace(
            ".osim", "_markers.osim")
        self._SCALETOOL_OUTPUT = self.m_DATA_PATH+"__scaleTool.xml"

    def preProcess(self, acq, staticFileNoExt):
        self.m_staticFileNoExt = staticFileNoExt
        btkTools.smartWriter( acq, self.m_DATA_PATH + self.m_staticFileNoExt[:-4], extension="trc")

        self.m_staticFile = self.m_DATA_PATH+staticFileNoExt + ".trc"

    def _timeRangeFromStatic(self):
        static = opensim.MarkerData(self.m_staticFile)
        initial_time = static.getStartFrameTime()
        final_time = static.getLastFrameTime()
        range_time = opensim.ArrayDouble()
        range_time.set(0, initial_time)
        range_time.set(1, final_time)
        return range_time

    def setAnthropometry(self, mass, height):  # , age):
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
