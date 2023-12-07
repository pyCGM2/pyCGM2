from typing import List, Tuple, Dict, Optional,Union

from pyCGM2.IMU import imu
from pyCGM2.Utils import files
from pyCGM2.Model.Opensim.interface import opensimInterface
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Model.Opensim.interface import opensimInterface

import btk
import opensim


class ImuPlacerXMLProcedure(object):
    """
    OpenSense IMU placer procedure for setting up and running IMU calibration with OpenSim.

    Args:
        DATA_PATH (str): Data path where the OpenSim model and results will be stored.
        genericOsimFile (str): Path and filename of the generic OpenSim model file (.osim).
    """
    
    def __init__(self, DATA_PATH:str,genericOsimFile:str):
        """
        Initializes the ImuPlacerXMLProcedure with the specified data path and generic OpenSim model file.
        """
        super(ImuPlacerXMLProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH

        self.m_osim = files.getFilename(genericOsimFile)
        self.m_osim_calibrated = self.m_osim[:-5]+"-ImuCalibrated"+".osim"
      

        files.copyPaste(genericOsimFile, self.m_DATA_PATH + self.m_osim)

        self.m_sensor_to_opensim_rotations = None
        self.m_imuMapper = {}

        self.m_osimInterface = opensimInterface.osimInterface(self.m_DATA_PATH, self.m_osim)

    def setSetupFile(self,imuPlacerToolFile:str):
        """
        Set the setup file for the IMU placer tool.

        Args:
            imuPlacerToolFile (str): Path and filename of the OpenSense IMU placer setup file.
        """

        self.m_imuPlacerTool = self.m_DATA_PATH + "__imuPlacerTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(imuPlacerToolFile,self.m_imuPlacerTool)


    def prepareImuMapper(self,imuMapperDict:Dict):
        """
        Prepare the IMU mapper, associating OpenSim model bodies with IMU instances.

        Args:
            imuMapperDict (Dict): Dictionary mapping OpenSim body names to IMU instances.
        """
        for key in imuMapperDict:
            if key not in self.m_osimInterface.getBodies():
                LOGGER.logger.error(f"[pyCGM2] the key {key} of your mapper is not a body of the osim file")
                raise 
                
        self.m_imuMapper.update(imuMapperDict)

    def prepareImu(self,osimBody:str,imuInstance:imu.Imu):
        """
        Assign an IMU instance to an OpenSim model body.

        Args:
            osimBody (str): Name of the body in the OpenSim model.
            imuInstance (imu.Imu): A pyCGM2 IMU instance.
        """

        if osimBody not in self.m_osimInterface.getBodies():
            LOGGER.logger.error(f"[pyCGM2] the key {osimBody} of your mapper is not a body of the osim file")
            raise 

        self.m_imuMapper.update({osimBody : imuInstance})


    def prepareOrientationFile(self,staticFilenameNoExt:str,freq:int,order:List=[0,1,2,3]):
        """
        Prepare the orientation file (STO file) from IMUs for static calibration.

        Args:
            staticFilenameNoExt (str): Name of the STO file to be created for static calibration.
            freq (int): Sampling frequency of the IMU data.
            order (List, optional): Quaternion reordering. Defaults to [0, 1, 2, 3].
        """

        self.m_staticFile = staticFilenameNoExt+".sto"

        imuStorage = opensimIO.ImuStorageFile(self.m_DATA_PATH, self.m_staticFile,freq)
        for key in self.m_imuMapper:
            imuStorage.setData(key,self.m_imuMapper[key].getQuaternions()[:, order])
        imuStorage.construct(static=True)

        self.m_sensorOrientationFile = self.m_staticFile

    def prepareBaseImu(self, segmentName:str, heading:str):
        """
        Set the base IMU and its heading axis for the calibration procedure.

        Args:
            segmentName (str): Segment name associated with the base IMU.
            heading (str): The heading axis for the base IMU.
        """

        self.m_base_imu_label=segmentName+"_imu"
        self.m_base_heading_axis=heading

    def prepareSensorToOpensimRotation(self,eulerAngles:List):
        """
        Prepare Euler angles to align sensors to the OpenSim global coordinate system.

        Args:
            eulerAngles (List): List of 3 Euler angle values in radians.
        """
        if len(eulerAngles) !=3:
            raise ("[pyCGM2] - eulerAngles must be a list of 3 values in radian")
        self.m_sensor_to_opensim_rotations = eulerAngles

    def prepareXml(self):
        """
        Prepare and update the XML associated with the IMU placer setup file.
        """

        self.xml.set_one("model_file", self.m_osim)
        self.xml.set_one("base_imu_label", self.m_base_imu_label)
        self.xml.set_one("base_heading_axis", self.m_base_heading_axis)
        self.xml.set_one("orientation_file_for_calibration", self.m_sensorOrientationFile)

        self.xml.set_one("output_model_file",  self.m_osim_calibrated)

        if self.m_sensor_to_opensim_rotations is not None:
            self.xml.set_one("sensor_to_opensim_rotations", ' '.join(str(e) for e in self.m_sensor_to_opensim_rotations)) 

    def run(self):
        """
        Run the IMU placer procedure using the prepared setup.
        """

        self.xml.update()

        imu_placer = opensim.IMUPlacer(self.m_imuPlacerTool)
        imu_placer.run()

        self.finalize()

    def finalize(self):
        """
        Finalize the procedure after running, including any post-processing steps.
        """
        pass