# -*- coding: utf-8 -*-
import os
from distutils import extension
from weakref import finalize
from pyCGM2.Utils import files
from pyCGM2.Tools import btkTools
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Model.Opensim.interface.procedures import opensimProcedures
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Model.Opensim.interface import opensimInterface

import btk
import opensim


class ImuPlacerXMLProcedure(object):
    def __init__(self, DATA_PATH,genericOsimFile):
    
         
        super(ImuPlacerXMLProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH

        self.m_osim = files.getFilename(genericOsimFile)
        files.copyPaste(genericOsimFile, self.m_DATA_PATH + self.m_osim)

        self.m_sensor_to_opensim_rotations = None
        self.m_imuMapper = dict()

        self.m_osimInterface = opensimInterface.osimInterface(self.m_DATA_PATH, self.m_osim)

    def setSetupFile(self,imuPlacerToolFile):

        self.m_imuPlacerTool = self.m_DATA_PATH + "__imuPlacerTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(imuPlacerToolFile,self.m_imuPlacerTool)


    def setImuMapper(self,imuMapperDict):
        for key in imuMapperDict:
            if key not in self.m_osimInterface.getBodies():
                LOGGER.logger.error(f"[pyCGM2] the key {key} of your mapper is not a body of the osim file")
                raise Exception (f"[pyCGM2] the key {key} of your mapper is not a body of the osim file") 
                
        self.m_imuMapper.update(imuMapperDict)

    def placeImu(self,osimBody,imuInstance):
        if osimBody not in self.m_osimInterface.getBodies():
            LOGGER.logger.error(f"[pyCGM2] the key {osimBody} of your mapper is not a body of the osim file")
            raise Exception (f"[pyCGM2] the key {osimBody} of your mapper is not a body of the osim file") 

        self.m_imuMapper.update({osimBody : imuInstance})


    def prepareOrientationFile(self,freq,order=[0,1,2,3]):

        imuStorage = opensimIO.ImuStorageFile(self.m_DATA_PATH, "placement_orientations.sto",freq)
        for key in self.m_imuMapper:
            imuStorage.setData(key,self.m_imuMapper[key].getQuaternions()[:, order])
        imuStorage.construct(static=True)

        self.m_sensorOrientationFile = "placement_orientations.sto"

    def setBaseImu(self, segmentName, heading):
         self.m_base_imu_label=segmentName+"_imu"
         self.m_base_heading_axis=heading

    def setSensorToOpensimRotation(self,eulerAngles):
        if len(eulerAngles) !=3:
            raise ("[pyCGM2] - eulerAngles must be a list of 3 values in radian")
        self.m_sensor_to_opensim_rotations = eulerAngles

    def prepareXml(self):

        self.xml.set_one("model_file", self.m_osim)
        self.xml.set_one("base_imu_label", self.m_base_imu_label)
        self.xml.set_one("base_heading_axis", self.m_base_heading_axis)
        self.xml.set_one("orientation_file_for_calibration", self.m_sensorOrientationFile)

        if self.m_sensor_to_opensim_rotations is not None:
            self.xml.set_one("sensor_to_opensim_rotations", ' '.join(str(e) for e in self.m_sensor_to_opensim_rotations)) 

    def run(self):


        self.xml.update()

        imu_placer = opensim.IMUPlacer(self.m_imuPlacerTool)
        imu_placer.run()

    def finalize(self):
        pass