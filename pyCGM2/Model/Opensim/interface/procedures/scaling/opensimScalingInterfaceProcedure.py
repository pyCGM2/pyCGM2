from typing import List, Tuple, Dict, Optional,Union,Any

from pyCGM2.Utils import files
from pyCGM2.Tools import btkTools
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures
import pyCGM2
LOGGER = pyCGM2.LOGGER

import btk

import opensim


class ScalingXmlProcedure(opensimProcedures.OpensimInterfaceXmlProcedure):
    """
    Procedure for scaling XML processing in OpenSim. It handles the preparation and processing
    of static trials and setup files for scaling in OpenSim.

    Args:
        DATA_PATH (str): The path to the data directory.
        mass (float): The mass of the subject.
        height (float): The height of the subject.
    """
    def __init__(self, DATA_PATH: str, mass: float, height: float):
        super(ScalingXmlProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH
        self.m_staticFile = None

        self.m_mass=mass
        self.m_height=height

    
    def prepareStaticTrial_fromBtkAcq(self, acq: btk.btkAcquisition, staticFileNoExt: str) -> None:
        """
        Prepares a static trial from a BTK acquisition object.

        Args:
            acq (btk.btkAcquisition): The BTK acquisition object.
            staticFileNoExt (str): The filename (without extension) for the static file.
        """
        self.m_staticFile = staticFileNoExt
        self._staticTrc = btkTools.smartWriter( acq, self.m_DATA_PATH + staticFileNoExt, extension="trc")

        static = opensim.MarkerData(self._staticTrc)
        self.m_initial_time = static.getStartFrameTime()
        self.m_final_time = static.getLastFrameTime()


    def setSetupFiles(self, genericOsimFile: str, markersetFile: str, scaleToolFile: str) -> None:
        """
        Sets up the required files for scaling procedure in OpenSim.

        Args:
            genericOsimFile (str): Path to the generic OpenSim file.
            markersetFile (str): Path to the markerset file.
            scaleToolFile (str): Path to the scale tool file.
        """

        self.m_osim = files.getFilename(genericOsimFile)
        files.copyPaste(genericOsimFile, self.m_DATA_PATH + self.m_osim)

        self.m_markerset =  files.getFilename(markersetFile)
        files.copyPaste(markersetFile, self.m_DATA_PATH + self.m_markerset)

        self.m_scaleTool = self.m_DATA_PATH + "ScaleToolSetup.xml"
        self.xml = opensimInterface.opensimXmlInterface(scaleToolFile, self.m_scaleTool)



    def prepareXml(self):
        """
        Prepares the XML for the scaling process.
        """

        self.xml.set_one("mass", str(self.m_mass))
        self.xml.set_one("height", str(self.m_height))

        self.xml.getSoup().find("ScaleTool").attrs["name"] = "ScaleTool"
        self.xml.set_one(["GenericModelMaker","model_file"],self.m_osim)
        self.xml.set_one(["GenericModelMaker","marker_set_file"],self.m_markerset)
        self.xml.set_many("time_range", str(self.m_initial_time) + " " + str(self.m_final_time))
        self.xml.set_many("marker_file", files.getFilename(self._staticTrc))
        self.xml.set_one(["MarkerPlacer","output_model_file"],self.m_staticFile+ "-ScaledModel.osim")
        self.xml.set_one(["MarkerPlacer","output_marker_file"], self.m_staticFile+ "-markerset.xml")


    def run(self):
        """
        Runs the scaling process.
        """
       
        self.xml.update()

        scale_tool = opensim.ScaleTool(self.m_scaleTool)
        scale_tool.run()

        self.m_osimModel_name = self.m_staticFile+ "-ScaledModel.osim"
        self.m_osimModel = opensim.Model(self.m_DATA_PATH + self.m_staticFile+ "-ScaledModel.osim")

        self.finalize()

    def finalize(self):
        """
        Finalizes the scaling process, including renaming and saving files.
        """
        files.renameFile(self.m_scaleTool,self.m_DATA_PATH + self.m_staticFile+ "-ScaleToolSetup.xml")    




class ScalingXmlCgmProcedure(ScalingXmlProcedure):
    """
    Procedure for scaling XML processing specific to the CGM model in OpenSim. 
    Extends ScalingXmlProcedure to include CGM model-specific configurations.

        Args:
        DATA_PATH (str): The path to the data directory.
        modelVersion (str): Version of the CGM model.
        mass (float): The mass of the subject.
        height (float): The height of the subject.

    """
    def __init__(self, DATA_PATH: str, modelVersion: str, mass: float, height: float):
        super(ScalingXmlCgmProcedure,self).__init__(DATA_PATH,mass, height)

        self.m_modelVersion = modelVersion.replace(".", "") if modelVersion is not None else "UnversionedModel"

        if self.m_modelVersion == "CGM23":  
            markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\markerset\\CGM23-markerset.xml"
            genericOsimFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\pycgm2-gait2392_simbody.osim"
            scaleToolFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23_scaleSetup_template.xml"


        if self.m_modelVersion == "CGM22":  
            markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\markerset\\CGM22-markerset.xml"
            genericOsimFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\pycgm2-gait2392_simbody.osim"
            scaleToolFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\setup\\CGM22_scaleSetup_template.xml"

        self.m_osim = files.getFilename(genericOsimFile)
        files.copyPaste(genericOsimFile, self.m_DATA_PATH + self.m_osim)

        self.m_markerset =  files.getFilename(markersetFile)
        files.copyPaste(markersetFile, self.m_DATA_PATH + self.m_markerset)

        self.m_scaleTool = self.m_DATA_PATH + self.m_modelVersion+"-ScaleToolSetup.xml"
        self.xml = opensimInterface.opensimXmlInterface(scaleToolFile, self.m_scaleTool)


    def prepareXml(self):
        """
        Prepares the XML for the scaling process specific to the CGM model.
        """

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
        """
        Runs the scaling process for the CGM model.
        """
        
        self.xml.update()

        scale_tool = opensim.ScaleTool(self.m_scaleTool)
        scale_tool.run()

        self.m_osimModel_name = self.m_staticFile+ "-" + self.m_modelVersion+"-ScaledModel.osim"
        self.m_osimModel = opensim.Model(self.m_DATA_PATH + self.m_staticFile+ "-" + self.m_modelVersion+"-ScaledModel.osim")

        self.finalize()

    def finalize(self):
        """
        Finalizes the scaling process for the CGM model, including renaming and saving files.
        """
        files.renameFile(self.m_scaleTool,self.m_DATA_PATH + self.m_staticFile+ "-"+self.m_modelVersion+"-ScaleToolSetup.xml")    