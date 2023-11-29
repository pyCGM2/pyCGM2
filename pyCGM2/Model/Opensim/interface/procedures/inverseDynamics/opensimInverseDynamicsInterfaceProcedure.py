from typing import List, Tuple, Dict, Optional,Union,Any

import pyCGM2; LOGGER = pyCGM2.LOGGER
# pyCGM2
import btk


from pyCGM2.Tools import  btkTools,opensimTools
from pyCGM2.Utils import files
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures

import opensim



class InverseDynamicsXmlProcedure(opensimProcedures.OpensimInterfaceXmlProcedure):
    """
    Provides a procedure for setting up and executing inverse dynamics analyses in OpenSim.

    This class handles the preparation of trials from BTK acquisitions, configuration of analysis 
    parameters, and execution of the inverse dynamics tool in OpenSim.

    Args:
        DATA_PATH (str): Path to the data directory.
        scaledOsimName (str): Name of the scaled OpenSim model file.
        resultsDirectory (Optional[str]): Directory for storing results.
    """
        
    def __init__(self, DATA_PATH: str, scaledOsimName: str, resultsDirectory: Optional[str]):
        super(InverseDynamicsXmlProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH
        self.m_osimName = DATA_PATH + scaledOsimName
        self.m_resultsDir = "" if resultsDirectory is None else resultsDirectory+"//"
        self.m_RES_PATH = self.m_DATA_PATH+self.m_resultsDir

        files.createDir(self.m_DATA_PATH+self.m_resultsDir)

        self.m_modelVersion=""

        self.m_acq=None

    def setSetupFiles(self, idToolTemplateFile: str, externalLoadTemplateFile: str) -> None:
        """
        Sets up the necessary files for inverse dynamics analysis.

        Args:
            idToolTemplateFile (str): Path to the Inverse Dynamics tool template file.
            externalLoadTemplateFile (str): Path to the external load template file.
        """

        self.m_idTool = self.m_DATA_PATH +  "-idTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(idToolTemplateFile,self.m_idTool)
        self.m_externalLoad = self.m_DATA_PATH  + "-externalLoad.xml"
        self.xml_load = opensimInterface.opensimXmlInterface(externalLoadTemplateFile,self.m_externalLoad)
        

    def prepareTrial_fromBtkAcq(self, acq: btk.btkAcquisition, dynamicFile: str, mfpa: Any, 
                                progressionAxis: str, forwardProgression: bool) -> None:
        """
        Prepares the trial data from a BTK acquisition for inverse dynamics analysis.

        Args:
            acq (btk.btkAcquisition): BTK acquisition object.
            dynamicFile (str): Name of the dynamic file.
            mfpa (Any): Parameter for foot reaction analysis (specific type depends on the context).
            progressionAxis (str): Axis of progression.
            forwardProgression (bool): Direction of forward progression.
        """
        self.m_dynamicFile =dynamicFile
        self.m_acq = acq
        self.m_mfpa = mfpa
        self.m_progressionAxis = progressionAxis
        self.m_forwardProgression = forwardProgression

        self.m_ff = self.m_acq.GetFirstFrame()
        self.m_freq = self.m_acq.GetPointFrequency()

        self.m_beginTime = 0
        self.m_endTime = (self.m_acq.GetLastFrame() - self.m_ff)/self.m_freq
        self.m_frameRange = [int((self.m_beginTime*self.m_freq)+self.m_ff),int((self.m_endTime*self.m_freq)+self.m_ff)] 

        opensimTools.footReactionMotFile(
            self.m_acq, self.m_DATA_PATH+self.m_resultsDir+"\\"+self.m_dynamicFile+"_grf.mot",
            self.m_progressionAxis,self.m_forwardProgression,mfpa = self.m_mfpa)


    def setFrameRange(self, begin: Optional[int], end: Optional[int]) -> None:
        """
        Sets the frame range for the inverse dynamics analysis.

        Args:
            begin (Optional[int]): The beginning frame number. If None, starts from the first frame.
            end (Optional[int]): The ending frame number. If None, ends at the last frame.
        
        """
        if self.m_acq is None:
            raise Exception(f"[pyCGM2] - no acquisition detected - trial preparation from a btk::Acquisition not done")
        self.m_beginTime = 0.0 if begin is None else (begin-self.m_ff)/self.m_freq
        self.m_endTime = (self.m_acq.GetLastFrame() - self.m_ff)/self.m_freq  if end is  None else (end-self.m_ff)/self.m_freq

    def setTimeRange(self, begin: float, end: float) -> None:
        """
        Sets the time range for the inverse dynamics analysis.

        Args:
            begin (float): The beginning time in seconds.
            end (float): The ending time in seconds.
        """
        self.m_beginTime = begin
        self.m_endTime = end


    def prepareXml(self):
        """
        Prepares the XML configuration for the Inverse Dynamics Tool in OpenSim.
        """
        self.xml.getSoup().find("InverseDynamicsTool").attrs["name"] = "InverseDynamics"
        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_RES_PATH+ self.m_dynamicFile+".mot")
        self.xml.set_one("output_gen_force_file", self.m_dynamicFile+"-inverse_dynamics.sto")
        self.xml.set_one("lowpass_cutoff_frequency_for_coordinates","6")

        self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))

        self.xml.set_one("time_range",str(self.m_beginTime) + " " + str(self.m_endTime))

        self.xml.set_one("results_directory",  self.m_resultsDir)

        self.xml_load.set_one("datafile", self.m_RES_PATH+ self.m_dynamicFile+"_grf.mot")
        

    def run(self):
        """
        Executes the inverse dynamics analysis using the configured XML files.
        """

        self.xml.update()
        self.xml_load.update()
        
        idTool = opensim.InverseDynamicsTool(self.m_idTool)
        idTool.run()

        self.finalize()
    
    def finalize(self):
        """
        Finalizes the process, including renaming and cleanup of the setup files.
        """
        files.renameFile(self.m_idTool, 
                    self.m_DATA_PATH + self.m_dynamicFile+ "-IDTool-setup.xml")
        files.renameFile(self.m_externalLoad,
             self.m_DATA_PATH + self.m_dynamicFile + "-externalLoad.xml")



class InverseDynamicsXmlCgmProcedure(InverseDynamicsXmlProcedure):
    """
    Specialized procedure for handling inverse dynamics analyses in OpenSim, specifically tailored for CGM models.

    This class extends InverseDynamicsXmlProcedure to include configurations and customizations necessary for CGM model versions.

    Args:
        DATA_PATH (str): Path to the data directory.
        scaledOsimName (str): Name of the scaled OpenSim model file.
        resultsDirectory (Optional[str]): Directory for storing results. If None, defaults to DATA_PATH.
        modelVersion (str): The version of the CGM model to be used.    
    """
        
    def __init__(self, DATA_PATH: str, scaledOsimName: str, resultsDirectory: Optional[str], modelVersion: str):
        super(InverseDynamicsXmlCgmProcedure,self).__init__(DATA_PATH,scaledOsimName,resultsDirectory)

        self.m_modelVersion = modelVersion.replace(".", "") if modelVersion is not None else "UnversionedModel"

        if self.m_modelVersion == "CGM23": 
            idToolTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-idToolSetup_template.xml"
            externalLoadFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\walk_grf.xml"

        if self.m_modelVersion == "CGM22": 
            idToolTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\setup\\CGM22-idToolSetup_template.xml"
            externalLoadFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\setup\\walk_grf.xml"

        self.m_idTool = self.m_DATA_PATH + self.m_modelVersion + "-idTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(idToolTemplateFile,self.m_idTool)
        self.m_externalLoad = self.m_DATA_PATH + self.m_modelVersion + "-externalLoad.xml"
        self.xml_load = opensimInterface.opensimXmlInterface(externalLoadFile,self.m_externalLoad)

    def prepareXml(self):
        """
        Prepares the XML configuration specific to the CGM model for the Inverse Dynamics Tool in OpenSim.

        Customizes the XML settings to align with the specifics of the CGM model version being used.
        """
        self.xml.getSoup().find("InverseDynamicsTool").attrs["name"] = self.m_modelVersion+"-InverseDynamics"
        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_RES_PATH+ self.m_dynamicFile+".mot")
        self.xml.set_one("output_gen_force_file", self.m_dynamicFile+"-"+self.m_modelVersion+"-inverse_dynamics.sto")
        self.xml.set_one("lowpass_cutoff_frequency_for_coordinates","6")

        self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))

        self.xml.set_one("time_range",str(self.m_beginTime) + " " + str(self.m_endTime))

        self.xml.set_one("results_directory",  self.m_resultsDir)


        self.xml_load.set_one("datafile", self.m_RES_PATH+ self.m_dynamicFile+"_grf.mot")
        

    def run(self):
        """
        Executes the inverse dynamics analysis using the configured XML files, specifically tailored for the CGM model.
        """

        self.xml.update()
        self.xml_load.update()
        
        idTool = opensim.InverseDynamicsTool(self.m_idTool)
        idTool.run()

        self.finalize()
    
    def finalize(self):
        """
        Finalizes the CGM model-specific inverse dynamics process, including renaming and cleanup of setup files.
        """
        
        files.renameFile(self.m_idTool, 
                    self.m_DATA_PATH + self.m_dynamicFile+ "-"+self.m_modelVersion + "-IDTool-setup.xml")
        files.renameFile(self.m_externalLoad,
             self.m_DATA_PATH + self.m_dynamicFile + "-"+self.m_modelVersion + "-externalLoad.xml")