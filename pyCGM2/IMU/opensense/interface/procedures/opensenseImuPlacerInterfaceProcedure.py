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

import btk
import opensim


class ImuPlacerXMLProcedure(object):
    def __init__(self, DATA_PATH):
    
         
        super(ImuPlacerXMLProcedure,self).__init__()

        self.m_DATA_PATH = DATA_PATH
        self.m_staticFile = None

    def setSetupFiles(self, genericOsimFile,imuPlacerToolFile):

        self.m_osim = files.getFilename(genericOsimFile)
        files.copyPaste(genericOsimFile, self.m_DATA_PATH + self.m_osim)

        self.m_imuPlacerTool = self.m_DATA_PATH + "__imuPlacerTool-setup.xml"
        self.xml = opensimInterface.opensimXmlInterface(imuPlacerToolFile,self.m_imuPlacerTool)

    def prepareOrientationFile(self,imuMapper):
        self.m_imuMapper = imuMapper

        imuStorage = opensimIO.ImuStorageFile(self.m_DATA_PATH, "placement_orientations.sto")
        for key in self.m_imuMapper:
            imuStorage.setData(key,self.m_imuMapper[key].getQuaternions())
        imuStorage.construct(static=True)

        self.m_sensorOrientationFile = "placement_orientations.sto"

    def setBaseImu(self, segmentName, heading):
         self.m_base_imu_label=segmentName+"_imu"
         self.m_base_heading_axis=heading


    def prepareXml(self):

        self.xml.set_one("model_file", self.m_osim)
        self.xml.set_one("base_imu_label", self.m_base_imu_label)
        self.xml.set_one("base_heading_axis", self.m_base_heading_axis)
        self.xml.set_one("orientation_file_for_calibration", self.m_sensorOrientationFile)


    def run(self):


        self.xml.update()

        imu_placer = opensim.IMUPlacer(self.m_imuPlacerTool)
        imu_placer.run()

    def finalize(self):
        pass