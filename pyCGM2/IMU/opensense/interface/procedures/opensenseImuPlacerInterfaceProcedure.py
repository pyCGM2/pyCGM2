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
    """opensense IMU placer procedure

        Args:
            DATA_PATH (str): data path
            genericOsimFile (str): path+filename of the osim file
    """
    
    def __init__(self, DATA_PATH:str,genericOsimFile:str):
        super(ImuPlacerXMLProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH

        self.m_osim = files.getFilename(genericOsimFile)
        self.m_osim_calibrated = self.m_osim[:-5]+"-ImuCalibrated"+".osim"
      

        files.copyPaste(genericOsimFile, self.m_DATA_PATH + self.m_osim)

        self.m_sensor_to_opensim_rotations = None
        self.m_imuMapper = {}

        self.m_osimInterface = opensimInterface.osimInterface(self.m_DATA_PATH, self.m_osim)

    def setSetupFile(self,imuPlacerToolFile:str):
        """set the setup file

        Args:
            imuPlacerToolFile (str): path+filename of the opensense imu placer setup file
        """

        self.m_imuPlacerTool = self.m_DATA_PATH + "__imuPlacerTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(imuPlacerToolFile,self.m_imuPlacerTool)


    def prepareImuMapper(self,imuMapperDict:Dict):
        """prepare the imuMapper 

        Args:
            imuMapperDict (dict): imu mapper

        """
        for key in imuMapperDict:
            if key not in self.m_osimInterface.getBodies():
                LOGGER.logger.error(f"[pyCGM2] the key {key} of your mapper is not a body of the osim file")
                raise 
                
        self.m_imuMapper.update(imuMapperDict)

    def prepareImu(self,osimBody:str,imuInstance:imu.Imu):
        """assign an imu instance to an osim body

        Args:
            osimBody (str): name of the body in the osim file
            imuInstance (imu.Imu): a pyCGM2 IMU instance

        """
        if osimBody not in self.m_osimInterface.getBodies():
            LOGGER.logger.error(f"[pyCGM2] the key {osimBody} of your mapper is not a body of the osim file")
            raise 

        self.m_imuMapper.update({osimBody : imuInstance})


    def prepareOrientationFile(self,staticFilenameNoExt:str,freq:int,order:List=[0,1,2,3]):
        """prepare the orientation file ( mot file) from Imus

        Args:
            motionFilenameNoExt (str): name of the mot file to create
            freq (int): frequency
            order (list, optional): quaternion reordering. Defaults to [0,1,2,3].
        """

        self.m_staticFile = staticFilenameNoExt+".sto"

        imuStorage = opensimIO.ImuStorageFile(self.m_DATA_PATH, self.m_staticFile,freq)
        for key in self.m_imuMapper:
            imuStorage.setData(key,self.m_imuMapper[key].getQuaternions()[:, order])
        imuStorage.construct(static=True)

        self.m_sensorOrientationFile = self.m_staticFile

    def prepareBaseImu(self, segmentName:str, heading:str):
         """ set the imu base and its heading axis
         """ 
         self.m_base_imu_label=segmentName+"_imu"
         self.m_base_heading_axis=heading

    def prepareSensorToOpensimRotation(self,eulerAngles:List):
        """ euler angle to pass to opensim global coordinate system

        Args:
            eulerAngles (list): list of 3 values in radian
        """
        if len(eulerAngles) !=3:
            raise ("[pyCGM2] - eulerAngles must be a list of 3 values in radian")
        self.m_sensor_to_opensim_rotations = eulerAngles

    def prepareXml(self):
        """ prepare-update and update the xml associated with the setup file"""

        self.xml.set_one("model_file", self.m_osim)
        self.xml.set_one("base_imu_label", self.m_base_imu_label)
        self.xml.set_one("base_heading_axis", self.m_base_heading_axis)
        self.xml.set_one("orientation_file_for_calibration", self.m_sensorOrientationFile)

        self.xml.set_one("output_model_file",  self.m_osim_calibrated)

        if self.m_sensor_to_opensim_rotations is not None:
            self.xml.set_one("sensor_to_opensim_rotations", ' '.join(str(e) for e in self.m_sensor_to_opensim_rotations)) 

    def run(self):
        """run the procedure """

        self.xml.update()

        imu_placer = opensim.IMUPlacer(self.m_imuPlacerTool)
        imu_placer.run()

        self.finalize()

    def finalize(self):
        """finalize after running """
        pass