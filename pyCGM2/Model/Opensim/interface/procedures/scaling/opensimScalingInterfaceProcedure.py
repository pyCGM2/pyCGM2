# -*- coding: utf-8 -*-
from distutils import extension
from pyCGM2.Utils import files
from pyCGM2.Tools import btkTools
from pyCGM2.Model.Opensim.interface import opensimInterfaceFilters
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures
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


class ScalingXMLProcedure(opensimProcedures.OpensimInterfaceXmlProcedure):
    def __init__(self, DATA_PATH, modelVersion, osimTemplateFile, markersetTemplateFile, scaleToolTemplateFile,
                local=False):
        
        super(ScalingXMLProcedure,self).__init__()

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


    def setStaticTrial(self, acq, staticFileNoExt):
        self._staticTrc = btkTools.smartWriter( acq, self.m_DATA_PATH + staticFileNoExt, extension="trc")

        static = opensim.MarkerData(self._staticTrc)
        self.m_initial_time = static.getStartFrameTime()
        self.m_final_time = static.getLastFrameTime()

    def setAnthropometry(self, mass, height):
        self.xml.set_one("mass", str(mass))
        self.xml.set_one("height", str(height))


    def _prepareXml(self):
        # self.xml.getSoup().GenericModelMaker.model_file.string = self.m_osim
        # self.xml.getSoup().GenericModelMaker.marker_set_file.string = self.m_markerset
        # self.xml.getSoup().MarkerPlacer.output_model_file.string =  self.m_modelVersion+"-ScaledModel.osim"
        self.xml.getSoup().find("ScaleTool").attrs["name"] = self.m_modelVersion+"-Scale"
        self.xml.set_one(["GenericModelMaker","model_file"],self.m_osim)
        self.xml.set_one(["GenericModelMaker","marker_set_file"],self.m_markerset)
        self.xml.set_many("time_range", str(self.m_initial_time) + " " + str(self.m_final_time))
        self.xml.set_many("marker_file", files.getFilename(self._staticTrc))
        self.xml.set_one(["MarkerPlacer","output_model_file"],self.m_modelVersion+"-ScaledModel.osim")


    def run(self):

        if self.m_autoXml: self._prepareXml()
        self.xml.update()

        scale_tool = opensim.ScaleTool(self.m_scaleTool)
        
        scale_tool.run()
        self.m_osimModel_name = self.m_modelVersion+"-ScaledModel.osim"
        self.m_osimModel = opensim.Model(self.m_DATA_PATH+self.m_modelVersion+"-ScaledModel.osim")
