from typing import List, Tuple, Dict, Optional,Union,Any

import os
import pyCGM2; LOGGER = pyCGM2.LOGGER

# pyCGM2
from pyCGM2.Tools import  btkTools,opensimTools
from pyCGM2.Utils import files
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures
from pyCGM2.Model.Opensim import opensimIO

import btk
try:
    import opensim
except:
    try:
        from pyCGM2 import opensim4 as opensim
    except:
        LOGGER.logger.error("[pyCGM2] opensim not found on your system")

class AnalysesXmlProcedure(opensimProcedures.OpensimInterfaceXmlProcedure):
    """
    Handles the setup and execution of various analyses in OpenSim using XML configurations.

    Args:
            DATA_PATH (str): Path to the data directory.
            scaledOsimName (str): Name of the scaled OpenSim model file.
            resultsDirectory (Optional[str]): Directory for storing results. If None, defaults to DATA_PATH.

    """

    def __init__(self, DATA_PATH: str, scaledOsimName: str, resultsDirectory: Optional[str]):
        super(AnalysesXmlProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH
        self.m_osimName = DATA_PATH + scaledOsimName
        self.m_modelVersion = ""
        self.m_resultsDir = "" if resultsDirectory is None else resultsDirectory+"//"
        
        self._externalLoadApplied = True

        self.m_RES_PATH = self.m_DATA_PATH+self.m_resultsDir
        files.createDir(self.m_RES_PATH)

        self.m_acq = None 

    def setSetupFiles(self, analysisToolTemplateFile: str, externalLoadTemplateFile: Optional[str]) -> None:
        """
        Sets up the file paths for the analysis tool and, optionally, the external load file.

        Args:
            analysisToolTemplateFile (str): Path to the analysis tool template file.
            externalLoadTemplateFile (Optional[str]): Path to the external load template file, if applicable.
        """
        self.m_idAnalyses = self.m_DATA_PATH + "__analysesTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(analysisToolTemplateFile,self.m_idAnalyses)
   
        if externalLoadTemplateFile is not None:
            self.m_externalLoad = self.m_DATA_PATH + "__externalLoad.xml"
            self.xml_load = opensimInterface.opensimXmlInterface(externalLoadTemplateFile,self.m_externalLoad)
        else: 
            self._externalLoadApplied= False

    def prepareTrial_fromMotFiles(self, kinematicMotFile: str, externalLoadDataFile: Optional[str]) -> None:
        """
        Prepares the trial data from MOT files for analysis.

        Args:
            kinematicMotFile (str): Name of the kinematic MOT file.
            externalLoadDataFile (Optional[str]): Name of the external load data file, if available.
        
        Raises:
            Exception: If the specified MOT file is not found.
        """
        self.m_dynamicFile = kinematicMotFile[:-4]

        if not os.path.isfile(self.m_RES_PATH+kinematicMotFile):
            
            raise Exception(f"[pyCGM2] - your mot file {kinematicMotFile} is not in the folder {self.m_RES_PATH}")

        motDf = opensimIO.OpensimDataFrame(self.m_RES_PATH, kinematicMotFile)
        self.m_beginTime = 0
        self.m_endTime = motDf.getDataFrame()["time"].iloc[-1]

        if externalLoadDataFile is not None:
            if not os.path.isfile(self.m_RES_PATH+externalLoadDataFile):
                raise Exception(f"[pyCGM2] - your mot file {externalLoadDataFile} is not in the folder {self.m_RES_PATH}")
            self.m_externalLoadDataFile = self.m_RES_PATH + "\\"+ externalLoadDataFile


    def prepareTrial_fromBtkAcq(self, acq: btk.btkAcquisition, dynamicFile: str, mfpa: Any, 
                                progressionAxis: str, forwardProgression: bool) -> None:
        """
        Prepares the trial data from a BTK acquisition for analysis.

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

        if self._externalLoadApplied:
            self.m_externalLoadDataFile = self.m_RES_PATH + self.m_dynamicFile+"_grf.mot"
            opensimTools.footReactionMotFile(
                self.m_acq, self.m_externalLoadDataFile,
                self.m_progressionAxis,self.m_forwardProgression,mfpa = self.m_mfpa)
        else:
            LOGGER.logger.info("Trial without force plate - No dynamic trials prepared ")

    def setFrameRange(self, begin: Optional[int], end: Optional[int]) -> None:
        """
        Sets the frame range for the analysis.

        Args:
            begin (Optional[int]): The beginning frame number. If None, starts from the first frame.
            end (Optional[int]): The ending frame number. If None, ends at the last frame.
        
        Raises:
            Exception: If no acquisition data is detected.
        """
        if self.m_acq is None:
            raise Exception(f"[pyCGM2] - no acquisition detected - trial preparation from a btk::Acquisition not done")
        self.m_beginTime = 0.0 if begin is None else (begin-self.m_ff)/self.m_freq
        self.m_endTime = (self.m_acq.GetLastFrame() - self.m_ff)/self.m_freq  if end is  None else (end-self.m_ff)/self.m_freq

    def setTimeRange(self, begin: float, end: float) -> None:
        """
        Sets the time range for the analysis.

        Args:
            begin (float): The beginning time in seconds.
            end (float): The ending time in seconds.
        """
        self.m_beginTime = begin
        self.m_endTime = end

    def prepareXml(self):
        """
        Prepares the XML configuration for the analysis in OpenSim.
        """

        # self.xml.set_one("model_file", self.m_dynamicFile+".mot")
        self.xml.getSoup().find("AnalyzeTool").attrs["name"] = self.m_dynamicFile+"-analyses"

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_RES_PATH+self.m_dynamicFile+".mot")
        self.xml.set_one("results_directory",  self.m_resultsDir)
        self.xml.set_one("initial_time",str(self.m_beginTime))
        self.xml.set_one("final_time",str(self.m_endTime))

        if self._externalLoadApplied:
            self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))
           
            self.xml_load.set_one("datafile", self.m_RES_PATH + self.m_dynamicFile+"_grf.mot")


    def run(self):
        """
        Executes the analysis using the configured XML files.
        """

        self.xml.update()
        if self._externalLoadApplied:
            self.xml_load.update()


        tool = opensim.AnalyzeTool(self.m_idAnalyses)
        tool.run()

        self.finalize()

    def finalize(self):
        """
        Finalizes the process, including renaming and cleanup of the setup files.
        """
        files.renameFile(self.m_idAnalyses, 
            self.m_DATA_PATH + self.m_dynamicFile+ "-analysesTool-setup.xml")

        if self._externalLoadApplied:
            files.renameFile(self.m_externalLoad,
                self.m_DATA_PATH + self.m_dynamicFile + "-externalLoad.xml")


class AnalysesXmlCgmProcedure(AnalysesXmlProcedure):
    """
    Specialized procedure for handling analyses in OpenSim, specifically tailored for CGM models.

    This class extends AnalysesXmlProcedure to include configurations and customizations necessary for CGM model versions in various analyses within OpenSim.

    Args:
        DATA_PATH (str): Path to the data directory.
        scaledOsimName (str): Name of the scaled OpenSim model file.
        resultsDirectory (Optional[str]): Directory for storing results. If None, defaults to DATA_PATH.
        modelVersion (str): The version of the CGM model to be used.

    """

    def __init__(self, DATA_PATH: str, scaledOsimName: str, resultsDirectory: Optional[str], modelVersion: str):
        super(AnalysesXmlCgmProcedure,self).__init__(DATA_PATH,scaledOsimName,resultsDirectory)

        self.m_modelVersion = modelVersion.replace(".", "") if modelVersion is not None else "UnversionedModel"

        if self.m_modelVersion == "CGM23":
            analysisToolTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23-muscleAnalysisSetup_template.xml"
            externalLoadTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\walk_grf.xml"

        if self.m_modelVersion == "CGM22":
            analysisToolTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\setup\\CGM22-muscleAnalysisSetup_template.xml"
            externalLoadTemplateFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\setup\\walk_grf.xml"


        self.m_idAnalyses = self.m_DATA_PATH + self.m_modelVersion + "-analysesTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(analysisToolTemplateFile,self.m_idAnalyses)
   
        if externalLoadTemplateFile is not None:
            self.m_externalLoad = self.m_DATA_PATH + self.m_modelVersion + "-externalLoad.xml"
            self.xml_load = opensimInterface.opensimXmlInterface(externalLoadTemplateFile,self.m_externalLoad)
        else: 
            self._externalLoadApplied= False

    def prepareXml(self):
        """
        Prepares the XML configuration for the analysis in OpenSim.
        """

        # self.xml.set_one("model_file", self.m_dynamicFile+".mot")
        self.xml.getSoup().find("AnalyzeTool").attrs["name"] = self.m_dynamicFile+"-"+self.m_modelVersion+"-analyses"

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_RES_PATH+self.m_dynamicFile+".mot")
        self.xml.set_one("results_directory",  self.m_resultsDir)
        self.xml.set_one("initial_time",str(self.m_beginTime))
        self.xml.set_one("final_time",str(self.m_endTime))

        if self._externalLoadApplied:
            self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))
           
            self.xml_load.set_one("datafile", self.m_externalLoadDataFile)

    def finalize(self):
        """
        Finalizes the process, including renaming and cleanup of the setup files.
        """
        files.renameFile(self.m_idAnalyses, 
            self.m_DATA_PATH + self.m_dynamicFile+ "-"+self.m_modelVersion + "-analysesTool-setup.xml")

        if self._externalLoadApplied:
            files.renameFile(self.m_externalLoad,
                self.m_DATA_PATH + self.m_dynamicFile + "-"+self.m_modelVersion + "-externalLoad.xml")


class AnalysesXmlCgmDrivenModelProcedure(AnalysesXmlCgmProcedure):
    """
    Specialized procedure for handling analyses in OpenSim using CGM-driven models.

    This class extends AnalysesXmlCgmProcedure to work with CGM-driven models, allowing for the specification and analysis of reference poses.

    Args:
        DATA_PATH (str): Path to the data directory.
        scaledOsimName (str): Name of the scaled OpenSim model file.
        resultsDirectory (Optional[str]): Directory for storing results. If None, defaults to DATA_PATH.
        modelVersion (str): The version of the CGM model to be used.
    """

    def __init__(self, DATA_PATH: str, scaledOsimName: str, resultsDirectory: Optional[str], modelVersion: str):
        super(AnalysesXmlCgmDrivenModelProcedure,self).__init__(DATA_PATH,scaledOsimName,resultsDirectory,modelVersion)

        self.m_acq = None
    
        self.m_refPose=None
        if self.m_modelVersion == "CGM23":
            self.m_refPose = opensimIO.OpensimDataFrame(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\", "referencePose.mot")
        
        elif self.m_modelVersion == "CGM22":
            self.m_refPose = opensimIO.OpensimDataFrame(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM22\\", "referencePose.mot")

        if self.m_refPose is not None:
            self.m_beginTime = 0
            self.m_endTime = self.m_refPose.getDataFrame()["time"].iloc[-1]

        self.m_poseLabel = ""
        self._externalLoadApplied = False
        

    def setPose(self, poseLabel: str, qi: Optional[Dict[str, Any]] = None) -> None:
        """
        Sets the pose for the CGM-driven model analysis.

        Args:
            poseLabel (str): Label for the pose being analyzed.
            qi (Optional[Dict[str, Any]]): Dictionary of joint angles to modify in the reference pose.
        """
        self.m_poseLabel = poseLabel

        if qi is not None:
            for key in qi:
                self.m_refPose.getDataFrame()[key] = qi[key]
                # self.m_refPose.getDataFrame()["hip_flexion_r"] = 90.0#np.deg2rad(90.0)
                # self.m_refPose.getDataFrame()["knee_flexion_r"] = -90.0#np.deg2rad(-90.0)
        self.m_refPose.save( outDir = self.m_DATA_PATH+self.m_resultsDir+"\\" , filename=poseLabel+".mot")
        
        self.m_poseLabel+".mot"

    def forceMotFile(self,motfile):
        self.m_poseLabel = motfile[:-4]

        files.copyPaste(self.m_DATA_PATH+ motfile,self.m_DATA_PATH+self.m_resultsDir+"\\"+motfile)
        self.m_refPose = opensimIO.OpensimDataFrame(self.m_DATA_PATH+self.m_resultsDir+"\\", motfile)

        self.m_beginTime = 0
        self.m_endTime = self.m_refPose.getDataFrame()["time"].iloc[-1]
    
    def prepareXml(self):
        """
        Prepares the XML configuration for the analysis in OpenSim.
        """

        # self.xml.set_one("model_file", self.m_dynamicFile+".mot")
        self.xml.getSoup().find("AnalyzeTool").attrs["name"] = self.m_modelVersion +"-Pose["+self.m_poseLabel+"]"

        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("coordinates_file", self.m_DATA_PATH+self.m_resultsDir+"\\"+self.m_poseLabel+".mot")
        self.xml.set_one("results_directory",  self.m_resultsDir)
        self.xml.set_one("initial_time",str(self.m_beginTime))
        self.xml.set_one("final_time",str(self.m_endTime))

        if self._externalLoadApplied:
            self.xml.set_one("external_loads_file", files.getFilename(self.m_externalLoad))
           
            self.xml_load.set_one("datafile", self.m_DATA_PATH+self.m_resultsDir + "\\"+ self.m_dynamicFile+"_grf.mot")


    def finalize(self):
        """
        Finalizes the process, including renaming and cleanup of the setup files.
        """
        files.renameFile(self.m_idAnalyses, 
            self.m_DATA_PATH + self.m_modelVersion + "-Pose["+self.m_poseLabel+"]-analysesTool-setup.xml")

        if self._externalLoadApplied:
            files.renameFile(self.m_externalLoad,
                self.m_DATA_PATH + self.m_modelVersion + "-Pose["+self.m_poseLabel+"]-externalLoad.xml")

