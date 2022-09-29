# -*- coding: utf-8 -*-
import os
from distutils import extension
from weakref import finalize
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
    def __init__(self, DATA_PATH,modelVersion):
    
         
        super(ScalingXMLProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH
        self.m_modelVersion = modelVersion.replace(".", "") if modelVersion is not None else "UnversionedModel"

        self.m_staticFile = None

    
    def setStaticTrial(self, acq, staticFileNoExt):
        self.m_staticFile = staticFileNoExt
        self._staticTrc = btkTools.smartWriter( acq, self.m_DATA_PATH + staticFileNoExt, extension="trc")

        static = opensim.MarkerData(self._staticTrc)
        self.m_initial_time = static.getStartFrameTime()
        self.m_final_time = static.getLastFrameTime()


    def setSetupFiles(self,genericOsimFile, markersetFile, scaleToolFile):

        self.m_osim = files.getFilename(genericOsimFile)
        files.copyPaste(genericOsimFile, self.m_DATA_PATH + self.m_osim)

        self.m_markerset =  files.getFilename(markersetFile)
        files.copyPaste(markersetFile, self.m_DATA_PATH + self.m_markerset)

        self.m_scaleTool = self.m_DATA_PATH + self.m_modelVersion+"-ScaleToolSetup.xml"
        self.xml = opensimInterfaceFilters.opensimXmlInterface(scaleToolFile, self.m_scaleTool)

    def setAnthropometry(self, mass, height):
        self.m_mass=mass
        self.m_height=height

    def setModelVersion(self, modelVersion):
        self.m_modelVersion = modelVersion.replace(".", "")

    def prepareXml(self):

        self.xml.set_one("mass", str(self.m_mass))
        self.xml.set_one("height", str(self.m_height))

        self.xml.getSoup().find("ScaleTool").attrs["name"] = self.m_modelVersion+"-Scale"
        self.xml.set_one(["GenericModelMaker","model_file"],self.m_osim)
        self.xml.set_one(["GenericModelMaker","marker_set_file"],self.m_markerset)
        self.xml.set_many("time_range", str(self.m_initial_time) + " " + str(self.m_final_time))
        self.xml.set_many("marker_file", files.getFilename(self._staticTrc))
        self.xml.set_one(["MarkerPlacer","output_model_file"],self.m_staticFile+ "-"+ self.m_modelVersion+"-ScaledModel.osim")
        self.xml.set_one(["MarkerPlacer","output_marker_file"], self.m_staticFile+ "-"+ self.m_modelVersion+"-markerset.xml")


    def run(self):

        
        self.xml.update()

        scale_tool = opensim.ScaleTool(self.m_scaleTool)
        scale_tool.run()

        self.m_osimModel_name = self.m_staticFile+ "-" + self.m_modelVersion+"-ScaledModel.osim"
        self.m_osimModel = opensim.Model(self.m_DATA_PATH + self.m_staticFile+ "-" + self.m_modelVersion+"-ScaledModel.osim")

        self.finalize()

    def finalize(self):
        files.renameFile(self.m_scaleTool,self.m_DATA_PATH + self.m_staticFile+ "-"+self.m_modelVersion+"-ScaleToolSetup.xml")    

        