# coding: utf-8
#pytest -s --disable-pytest-warnings  test_opensense.py::Test_Opensense::test_opensenseProcessing_blueTrident_pyCGM2gait2392model
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_BlueTridentOrientation::test_relativeAngles
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_Garches::test_reader
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_captureU::test_captureUscript
# from __future__ import unicode_literals

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pyCGM2
LOGGER = pyCGM2.LOGGER



from pyCGM2.Tools import btkTools
from pyCGM2.IMU import imu
from pyCGM2.Utils import files

from pyCGM2.IMU import imuFilters
from pyCGM2.IMU.Procedures import imuReaderProcedures
from pyCGM2.IMU.Procedures import relativeImuAngleProcedures
from pyCGM2.IMU.Procedures import imuMotionProcedure

from pyCGM2.Math import pose


from pyCGM2.IMU.opensense.interface import opensenseFilters
from pyCGM2.IMU.opensense.interface.procedures import opensenseImuPlacerInterfaceProcedure
from pyCGM2.IMU.opensense.interface.procedures import opensenseImuKinematicFitterProcedure
from pyCGM2.Model.Opensim import opensimIO

class Test_Opensense:

    def test_opensenseProcessing_blueTrident_Rajagopal2015(self):

        data_path = pyCGM2.TEST_DATA_PATH + "Opensense\\nexus\\"
        os.chdir(data_path)

        staticFilename = "Calibration.c3d"

        sensorToOpensim = [-np.pi/2,0,0]

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"1")
        irf = imuFilters.ImuReaderFilter(irp)
        imu1 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"2")
        irf = imuFilters.ImuReaderFilter(irp)
        imu2 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"3")
        irf = imuFilters.ImuReaderFilter(irp)
        imu3 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"4")
        irf = imuFilters.ImuReaderFilter(irp)
        imu4 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"5")
        irf = imuFilters.ImuReaderFilter(irp)
        imu5 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"6")
        irf = imuFilters.ImuReaderFilter(irp)
        imu6 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"7")
        irf = imuFilters.ImuReaderFilter(irp)
        imu7 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"8")
        irf = imuFilters.ImuReaderFilter(irp)
        imu8 = irf.run() 

        
        
        freq= imu1.m_freq
        

        osimTemplateFullFile = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\Settings\\opensense\\Rajagopal2015_opensense.osim"
        imuPlacerToolFullFile = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\Settings\\opensense\\imuPlacer_Setup.xml"
        
        proc = opensenseImuPlacerInterfaceProcedure.ImuPlacerXMLProcedure(data_path,osimTemplateFullFile)
        proc.setSetupFile(imuPlacerToolFullFile)
        #proc.setImuMapper( {"pelvis2":imu1,
        #                  "femur_l":imu2,
        #                  "femur_r":imu3,
        #                  "tibia_l":imu4,
        #                  "tibia_r":imu5,
        #                  "calcn_l":imu6,
        #                  "calcn_r":imu7,
        #                  "torso":imu8})
        proc.placeImu("pelvis",imu1)
        proc.placeImu("femur_l",imu2)
        proc.placeImu("femur_r",imu3)
        proc.placeImu("tibia_l",imu4)
        proc.placeImu("tibia_r",imu5)
        proc.placeImu("calcn_l",imu6)
        proc.placeImu("calcn_r",imu7)
        proc.placeImu("torso",imu8)
        proc.prepareOrientationFile(staticFilename[:-4],freq,order=[3,0,1,2])
        proc.setBaseImu("pelvis","-z")
        proc.setSensorToOpensimRotation(sensorToOpensim)
        proc.prepareXml()

        filter = opensenseFilters.opensenseInterfaceImuPlacerFilter(proc)
        filter.run()


        
        #----- motion trial------------

        dynamicFile = "Walk.c3d"

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"1")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu1 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"2")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu2 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"3")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu3 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"4")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu4 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"5")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu5 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"6")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu6 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"7")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu7 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"7")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu7 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"8")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu8 = irf.run() 



        imuInverseKinematicToolFullFile = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\Settings\\opensense\\imuInverseKinematics_Setup.xml"

        calibratedModel = "Rajagopal_2015_calibrated.osim"

        freq = imu1.m_freq

        procIk = opensenseImuKinematicFitterProcedure.ImuInverseKinematicXMLProcedure(data_path,calibratedModel,"resutsTest")
        procIk.setSetupFile(imuInverseKinematicToolFullFile)
        procIk.setImuMapper( {"pelvis":imu1,
                         "femur_l":imu2,
                         "femur_r":imu3,
                         "tibia_l":imu4,
                         "tibia_r":imu5,
                         "calcn_l":imu6,
                         "calcn_r":imu7,
                          "torso":imu8 })
        procIk.prepareOrientationFile(dynamicFile[:-4],freq,order=[3,0,1,2])
        procIk.setSensorToOpensimRotation(sensorToOpensim)
        procIk.prepareXml()

        filter = opensenseFilters.opensenseInterfaceImuInverseKinematicFilter(procIk)
        filter.run()

    def test_opensenseProcessing_blueTrident_pyCGM2gait2392model(self):

        data_path = pyCGM2.TEST_DATA_PATH + "Opensense\\nexus\\"
        os.chdir(data_path)

        staticFilename = "Calibration.c3d"

        sensorToOpensim = [-np.pi/2,0,0]

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"1")
        irf = imuFilters.ImuReaderFilter(irp)
        imu1 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"2")
        irf = imuFilters.ImuReaderFilter(irp)
        imu2 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"3")
        irf = imuFilters.ImuReaderFilter(irp)
        imu3 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"4")
        irf = imuFilters.ImuReaderFilter(irp)
        imu4 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"5")
        irf = imuFilters.ImuReaderFilter(irp)
        imu5 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"6")
        irf = imuFilters.ImuReaderFilter(irp)
        imu6 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"7")
        irf = imuFilters.ImuReaderFilter(irp)
        imu7 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+staticFilename,"8")
        irf = imuFilters.ImuReaderFilter(irp)
        imu8 = irf.run() 

        
        
        freq= imu1.m_freq
        

        osimTemplateFullFile = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\Settings\\opensim\\interface\\CGM23\\pycgm2-gait2392_simbody.osim"
        imuPlacerToolFullFile = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\Settings\\opensense\\imuPlacer_Setup.xml"
        
        proc = opensenseImuPlacerInterfaceProcedure.ImuPlacerXMLProcedure(data_path,osimTemplateFullFile)
        proc.setSetupFile(imuPlacerToolFullFile)
        #proc.setImuMapper( {"pelvis2":imu1,
        #                  "femur_l":imu2,
        #                  "femur_r":imu3,
        #                  "tibia_l":imu4,
        #                  "tibia_r":imu5,
        #                  "calcn_l":imu6,
        #                  "calcn_r":imu7,
        #                  "torso":imu8})
        proc.placeImu("pelvis",imu1)
        proc.placeImu("femur_l",imu2)
        proc.placeImu("femur_r",imu3)
        proc.placeImu("tibia_l",imu4)
        proc.placeImu("tibia_r",imu5)
        proc.placeImu("calcn_l",imu6)
        proc.placeImu("calcn_r",imu7)
        proc.placeImu("torso",imu8)
        proc.prepareOrientationFile(staticFilename[:-4],freq,order=[3,0,1,2])
        proc.setBaseImu("pelvis","-z")
        proc.setSensorToOpensimRotation(sensorToOpensim)
        proc.prepareXml()

        filter = opensenseFilters.opensenseInterfaceImuPlacerFilter(proc)
        filter.run()


        
        #----- motion trial------------

        dynamicFile = "Walk.c3d"

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"1")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu1 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"2")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu2 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"3")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu3 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"4")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu4 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"5")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu5 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"6")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu6 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"7")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu7 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"7")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu7 = irf.run() 

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"8")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu8 = irf.run() 



        imuInverseKinematicToolFullFile = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\Settings\\opensense\\imuInverseKinematics_Setup.xml"

        calibratedModel = "Rajagopal_2015_calibrated.osim"

        freq = imu1.m_freq

        procIk = opensenseImuKinematicFitterProcedure.ImuInverseKinematicXMLProcedure(data_path,calibratedModel,"resutsTest")
        procIk.setSetupFile(imuInverseKinematicToolFullFile)
        procIk.setImuMapper( {"pelvis":imu1,
                         "femur_l":imu2,
                         "femur_r":imu3,
                         "tibia_l":imu4,
                         "tibia_r":imu5,
                         "calcn_l":imu6,
                         "calcn_r":imu7,
                          "torso":imu8 })
        procIk.prepareOrientationFile(dynamicFile[:-4],freq,order=[3,0,1,2])
        procIk.setSensorToOpensimRotation(sensorToOpensim)
        procIk.prepareXml()

        filter = opensenseFilters.opensenseInterfaceImuInverseKinematicFilter(procIk)
        filter.run()