from typing import List, Tuple, Dict, Optional,Union

from pyCGM2.Utils import files
from pyCGM2.IMU import imu
from pyCGM2.Model.Opensim.interface import opensimInterface
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Model.Opensim import opensimIO

import opensim


class ImuInverseKinematicXMLProcedure(object):
    def __init__(self, DATA_PATH:str,calibratedOsimName:str, resultsDirectory:str):
        """procedure to run opensense Inverse Kinematics

        Args:
            DATA_PATH (str): data path
            calibratedOsimName (str): name of the osim file
            resultsDirectory (str): results directory
        """
        super(ImuInverseKinematicXMLProcedure,self).__init__()
        

        self.m_DATA_PATH = DATA_PATH

        self.m_resultsDir = "" if resultsDirectory is None else resultsDirectory

        files.createDir(self.m_DATA_PATH+self.m_resultsDir) # required to save the mot file. (opensim issue ?) 

        self.m_osimName = DATA_PATH + calibratedOsimName
        self.m_osimInterface = opensimInterface.osimInterface(self.m_DATA_PATH, calibratedOsimName)

        self.m_accuracy = 1e-8
        self.m_imuMapper = {}

        self.m_sensor_to_opensim_rotations = None

    def setSetupFile(self,imuInverseKinematicToolFile:str):
        """set the inverse kinematics setup file

        Args:
            imuInverseKinematicToolFile (str): path+filename of the opensense setupfile
        """

        self.m_imuInverseKinematicTool = self.m_DATA_PATH + "__imuInverseKinematics_Setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(imuInverseKinematicToolFile,self.m_imuInverseKinematicTool)

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

    def prepareOrientationFile(self,motionFilenameNoExt:str,freq:int,order:List=[0,1,2,3]):
        """prepare the orientation file ( mot file) from Imus

        Args:
            motionFilenameNoExt (str): name of the mot file to create
            freq (int): frequency
            order (list, optional): quaternion reordering. Defaults to [0,1,2,3].
        """
        self.m_dynamicFile = motionFilenameNoExt+".sto"

        imuStorage = opensimIO.ImuStorageFile(self.m_DATA_PATH, self.m_dynamicFile,freq)
        for key in self.m_imuMapper:
            imuStorage.setData(key,self.m_imuMapper[key].getQuaternions()[:, order])
        imuStorage.construct(static=False)

        self.m_sensorOrientationFile = self.m_dynamicFile

        opensimDf = opensimIO.OpensimDataFrame(self.m_DATA_PATH,self.m_dynamicFile)
        self.m_beginTime = opensimDf.m_dataframe["time"].iloc[0]
        self.m_endTime = opensimDf.m_dataframe["time"].iloc[-1]


    def setTimeRange(self,beginTime:Optional[float],lastTime:Optional[float]):
        """set begin and end time

        Args:
            beginTime (Optional[float]): start
            lastTime (Optional[float]): end
        """
        if beginTime is not None and beginTime <-self.m_endTime: 
            self.m_beginTime = beginTime
        if lastTime is not None and lastTime <-self.m_endTime: 
            self.m_endTime = lastTime

    def prepareSensorToOpensimRotation(self,eulerAngles:list):
        """ euler angle to pass to opensim global coordinate system

        Args:
            eulerAngles (list): list of 3 values in radian
        """
        if len(eulerAngles) !=3:
            LOGGER.logger.error(f"eulerAngles must be a list of 3 values in radian")
            raise 
        self.m_sensor_to_opensim_rotations = eulerAngles

    def prepareXml(self):
        """ prepare-update and update the xml associated with the setup file"""
        self.xml.set_one("model_file", self.m_osimName)
        self.xml.set_one("orientations_file", self.m_dynamicFile)


        self.xml.set_one("output_motion_file", self.m_DATA_PATH+self.m_resultsDir + "\\"+ self.m_dynamicFile[:-4]+".mot")
        # for marker in self.m_weights.keys():
        #     self.xml.set_inList_fromAttr("IKMarkerTask","weight","name",marker,str(self.m_weights[marker]))
        # self.xml.set_one("accuracy",str(self.m_accuracy))
        self.xml.set_one("time_range",str(self.m_beginTime) + " " + str(self.m_endTime))
        self.xml.set_one("results_directory",  self.m_resultsDir)

        if self.m_sensor_to_opensim_rotations is not None:
            self.xml.set_one("sensor_to_opensim_rotations", ' '.join(str(e) for e in self.m_sensor_to_opensim_rotations)) 



    def run(self):
        """run the procedure """

        self.xml.update()
        
        imu_ikTool = opensim.IMUInverseKinematicsTool(self.m_imuInverseKinematicTool)
        imu_ikTool.run()

        self.finalize()

    def finalize(self):
        """finalize after running """
        # rename the xml setup file with the filename as suffix
        files.renameFile(self.m_imuInverseKinematicTool, 
                    self.m_DATA_PATH + self.m_dynamicFile[:-3] + "-imuInverseKinematicTool-setup.xml")