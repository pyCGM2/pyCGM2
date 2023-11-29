from typing import List, Tuple, Dict, Optional,Union,Any

import pyCGM2; LOGGER = pyCGM2.LOGGER

# pyCGM2
from pyCGM2.Tools import  opensimTools
from pyCGM2.Utils import files
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures

import btk
import opensim



class StaticOptimisationXmlProcedure(opensimProcedures.OpensimInterfaceXmlProcedure):
    """Handles the setup and execution of static optimization analyses in OpenSim.

    This class is responsible for configuring the Static Optimization tool in OpenSim based on
    the provided BTK acquisition data and specified model files.

    Args:
        DATA_PATH (str): Path to the data directory.
        scaledOsimName (str): Name of the scaled OpenSim model file.
        resultsDirectory (Optional[str]): Directory for storing results. If None, defaults to DATA_PATH.
    """

    def __init__(self, DATA_PATH: str, scaledOsimName: str, resultsDirectory: Optional[str]):
        super(StaticOptimisationXmlProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH
        self.m_osimName = DATA_PATH + scaledOsimName

        self.m_resultsDir = "" if resultsDirectory is None else resultsDirectory+"//"
        self.m_RES_PATH = self.m_DATA_PATH+self.m_resultsDir

        files.createDir(self.m_DATA_PATH+self.m_resultsDir)
        self.m_modelVersion = ""
        self.m_acq=None

    def setSetupFiles(self, analysisToolTemplateFile: str, externalLoadTemplateFile: str) -> None:
        """
        Sets up the file paths for the Static Optimization tool and external load file.

        Args:
            analysisToolTemplateFile (str): Path to the Static Optimization tool template file.
            externalLoadTemplateFile (str): Path to the external load template file.
        """

        self.m_soTool = self.m_DATA_PATH + "__SOTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(analysisToolTemplateFile,self.m_soTool)
   
        self.m_externalLoad = self.m_DATA_PATH + "__externalLoad.xml"
        self.xml_load = opensimInterface.opensimXmlInterface(externalLoadTemplateFile,self.m_externalLoad)
    

    def setResultsDirname(self, dirname: str) -> None:
        """
        Sets the results directory name.

        Args:
            dirname (str): The name of the directory to store results.
        """
        self.m_resultsDir = dirname    

    def prepareTrial_fromBtkAcq(self, acq: btk.btkAcquisition, dynamicFile: str, mfpa: Any, 
                                progressionAxis: str, forwardProgression: bool) -> None:
        """
        Prepares the trial data from a BTK acquisition for static optimization analysis.

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
        Sets the frame range for the static optimization analysis.

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
        Sets the time range for the static optimization analysis.

        Args:
            begin (float): The beginning time in seconds.
            end (float): The ending time in seconds.
        """
        self.m_beginTime = begin
        self.m_endTime = end


    def prepareXml(self):
        """
        Prepares the XML configuration for the Static Optimization Tool in OpenSim.
        """

        # self.xml.set_one("model_file", self.m_dynamicFile+".mot")
        self.xml.getSoup().find("AnalyzeTool").attrs["name"] = self.m_dynamicFile+"-analyses"

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_RES_PATH+ self.m_dynamicFile+".mot")
        self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))

        self.xml.set_one("results_directory",  self.m_resultsDir)

        self.xml.set_one("initial_time",str(self.m_beginTime))
        self.xml.set_one("final_time",str(self.m_endTime))

        self.xml.m_soup.AnalysisSet.start_time.string =  str(self.m_beginTime)
        self.xml.m_soup.AnalysisSet.end_time.string =  str(self.m_endTime)

        self.xml_load.set_one("datafile", self.m_RES_PATH+ self.m_dynamicFile+"_grf.mot")


    def run(self):
        """
        Executes the static optimization analysis using the configured XML files.
        """
        self.xml.update()
        self.xml_load.update()

        tool = opensim.AnalyzeTool(self.m_soTool)
        tool.run()

        self.finalize()

    def finalize(self):
        """
        Finalizes the process, including renaming and cleanup of the setup files.
        """
        files.renameFile(self.m_soTool, 
            self.m_DATA_PATH + self.m_dynamicFile+ "-SOTool-setup.xml")
        files.renameFile(self.m_externalLoad,
             self.m_DATA_PATH + self.m_dynamicFile + "-externalLoad.xml")


class StaticOptimisationXmlCgmProcedure(StaticOptimisationXmlProcedure):
    """
    Specialized procedure for handling static optimization analyses in OpenSim, specifically tailored for CGM models.

    This class extends StaticOptimisationXmlProcedure to include configurations and customizations necessary for CGM model versions in static optimization analyses.

    Args:
        DATA_PATH (str): Path to the data directory.
        scaledOsimName (str): Name of the scaled OpenSim model file.
        modelVersion (str): The version of the CGM model to be used.
        resultsDirectory (Optional[str]): Directory for storing results. If None, defaults to DATA_PATH.

    """

    def __init__(self, DATA_PATH: str, scaledOsimName: str, modelVersion: str, resultsDirectory: Optional[str]):
        super(StaticOptimisationXmlCgmProcedure,self).__init__(DATA_PATH,scaledOsimName,resultsDirectory)

        self.m_modelVersion = modelVersion.replace(".", "") if modelVersion is not None else "UnversionedModel"

        if self.m_modelVersion == "CGM23":
            analysisToolTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\setup\\CGM23\\CGM23-soSetup_template.xml"
            externalLoadTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\setup\\walk_grf.xml"

        if self.m_modelVersion == "CGM22":
            analysisToolTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\setup\\CGM22\\CGM22-soSetup_template.xml"
            externalLoadTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\setup\\walk_grf.xml"


        self.m_soTool = self.m_DATA_PATH + self.m_modelVersion + "-SOTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(analysisToolTemplateFile,self.m_soTool)
   
        self.m_externalLoad = self.m_DATA_PATH + self.m_modelVersion + "-externalLoad.xml"
        self.xml_load = opensimInterface.opensimXmlInterface(externalLoadTemplateFile,self.m_externalLoad)
    
   
    def prepareXml(self):
        """
        Prepares the XML configuration for the Static Optimization Tool in OpenSim.
        """
        # self.xml.set_one("model_file", self.m_dynamicFile+".mot")
        self.xml.getSoup().find("AnalyzeTool").attrs["name"] = self.m_dynamicFile+"-"+self.m_modelVersion+"-analyses"

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_RES_PATH+ self.m_dynamicFile+".mot")
        self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))

        self.xml.set_one("results_directory",  self.m_resultsDir)

        self.xml.set_one("initial_time",str(self.m_beginTime))
        self.xml.set_one("final_time",str(self.m_endTime))

        self.xml.m_soup.AnalysisSet.start_time.string =  str(self.m_beginTime)
        self.xml.m_soup.AnalysisSet.end_time.string =  str(self.m_endTime)

        self.xml_load.set_one("datafile", self.m_RES_PATH+ self.m_dynamicFile+"_grf.mot")


    def finalize(self):
        """
        Finalizes the process, including renaming and cleanup of the setup files.
        """
        
        files.renameFile(self.m_soTool, 
            self.m_DATA_PATH + self.m_dynamicFile+ "-"+self.m_modelVersion + "-SOTool-setup.xml")
        files.renameFile(self.m_externalLoad,
             self.m_DATA_PATH + self.m_dynamicFile + "-"+self.m_modelVersion + "-externalLoad.xml")