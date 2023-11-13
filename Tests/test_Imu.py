# coding: utf-8
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_BlueTrident::test_reader_csv
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_BlueTridentOrientation::test_relativeAngles
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_Garches::test_reader
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_captureU::test_captureUscript
# from __future__ import unicode_literals

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

from viconnexusapi import ViconUtils

def Vicon_practice_GlobalAngle(file_name):
    raw_data = pd.read_csv(file_name)

    # Extract the quaternion data

    trial_quaternion = np.transpose(np.array([raw_data.qx, raw_data.qy, raw_data.qz, raw_data.qr]))


    # Calculate the Direct Cosine Matrix from a quaternion input

    trial_helical = np.array([ViconUtils.AngleAxisFromQuaternion(trial_quaternion[frame])
                                        for frame in range(len(trial_quaternion))])
    trial_DCM = [ViconUtils.RotationMatrixFromAngleAxis(trial_helical[frame]) for frame in range(len(trial_helical))]

    # Invert Initial Orientation (i.e. Find transform to new LCS)
    # Or find one within first 100 frames if we have doubts about first frame?

    trial_initial_DCM = np.linalg.inv(trial_DCM[0])

    # Recalculate angles based on new Coordinate System

    trial_DCM_new = [np.matmul(trial_initial_DCM, trial_DCM[frame]) for frame in range(len(trial_DCM))]
    trial_helical_new = np.array([ViconUtils.AngleAxisFromMatrix(trial_DCM_new[frame])
                                for frame in range(len(trial_DCM_new))])

    # Convert back to degrees
    trial_new_deg = np.rad2deg(trial_helical_new)

    # # Plot trial_new_deg and trial_helical
    # plot1 = plt.figure(1)
    # plt.plot(trial_new_deg[:, 0], "-r", label="x")
    # plt.plot(trial_new_deg[:, 1], "-g", label="y")
    # plt.plot(trial_new_deg[:, 2], "-b", label="z")
    # plt.xlabel("Frames", loc='right')
    # plt.ylabel("Angle (degrees)")
    # plt.title("Re-aligned Orientation")
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #         fancybox=True, shadow=True, ncol=5)

    # plot2 = plt.figure(2)
    # plt.plot(np.rad2deg(trial_helical[:, 0]), "-r", label="x")
    # plt.plot(np.rad2deg(trial_helical[:, 1]), "-g", label="y")
    # plt.plot(np.rad2deg(trial_helical[:, 2]), "-b", label="z")
    # plt.xlabel("Frames", loc='right')
    # plt.ylabel("Angle (degrees)")
    # plt.title("Original Orientation")
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #         fancybox=True, shadow=True, ncol=5)
    # plt.show()

    return trial_new_deg, trial_helical



class Test_ImuReaders:

    def test_blueTridentAlignedCsv(self):
        fullfilename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\static_2sensors_csvFiles\\S1-1_TS-01436_2023-08-01-15-59-57_aligned.csv"
       
        imuTranslators  = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER +"IMU\\","viconBlueTrident.translators")
       
        irp = imuReaderProcedures.CsvProcedure(fullfilename, imuTranslators["Translators"] )
        irf = imuFilters.ImuReaderFilter(irp)
        imu1 = irf.run()
        
        irp2 = imuReaderProcedures.CsvProcedure(fullfilename, imuTranslators["Translators"] )
        irp2.downsample(100)
        irf = imuFilters.ImuReaderFilter(irp2)
        imu2 = irf.run()


    def test_blueTridentc3d(self):
        fullfilename = pyCGM2.TEST_DATA_PATH + "IMU\\angleMeasurement\\goniometer\\right36 -0to120 trial 01.c3d"

        # imuTranslators  = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER +"IMU\\","viconBlueTrident.translators")
       
        irp = imuReaderProcedures.C3dBlueTridentProcedure(fullfilename,"1")
        irf = imuFilters.ImuReaderFilter(irp)
        imu1 = irf.run()


        irp = imuReaderProcedures.C3dBlueTridentProcedure(fullfilename,"1")
        irp.downsample(100)
        irf = imuFilters.ImuReaderFilter(irp)
        imu1 = irf.run()

    def test_blueTridentNotAlignedCsv(self):
        fullfilename1 = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\Rouling Maxence_TS-01436_2022-04-26-16-34-56_lowg.csv"
        fullfilename2 = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\Rouling Maxence_TS-02122_2022-04-26-16-34-56_lowg.csv"
        fullfilename3 = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\Rouling Maxence_TS-02374_2022-04-26-16-34-56_lowg.csv"

        imuTranslators  = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER +"IMU\\","viconBlueTrident.translators")

        dataframes = imuReaderProcedures.synchroniseNotAlignedCsv([fullfilename1,fullfilename2,fullfilename3],timeColumn ="time_s")


        irp = imuReaderProcedures.DataframeProcedure(dataframes[0], imuTranslators["Translators"] )
        irf = imuFilters.ImuReaderFilter(irp)
        imu1 = irf.run()


        irp2 = imuReaderProcedures.DataframeProcedure(dataframes[0], imuTranslators["Translators"] )
        irp2.downsample(100)
        irf = imuFilters.ImuReaderFilter(irp2)
        imu2 = irf.run()
        



class Test_ImuMotion:
    def test_relativeAngles(self):
    
        fullfilename = pyCGM2.TEST_DATA_PATH + "IMU\\angleMeasurement\\goniometer\\right36 -0to120 trial 01.c3d"

        # imuTranslators  = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER +"IMU\\","viconBlueTrident.translators")
       
        irp = imuReaderProcedures.C3dBlueTridentProcedure(fullfilename,"1")
        irf = imuFilters.ImuReaderFilter(irp)
        imu1 = irf.run()        

        irp = imuReaderProcedures.C3dBlueTridentProcedure(fullfilename,"2")
        irf = imuFilters.ImuReaderFilter(irp)
        imu2 = irf.run()        


        # return the angle from euler sequence
        proc = relativeImuAngleProcedures.RelativeAnglesProcedure(representation = "Euler", eulerSequence="ZYX")
        filt =  imuFilters.ImuRelativeAnglesFilter(imu1,imu2, proc)
        jointFinalValues = filt.run()


        jointFinalValues = np.rad2deg(jointFinalValues)
        plt.figure()
        plt.plot(jointFinalValues[:,0],"-r")
        plt.plot(jointFinalValues[:,1],"-g")
        plt.plot(jointFinalValues[:,2],"-b")

        plt.show()


class Test_Vicon:
    def test_Vicon_practice_GlobalAngle(self):
  
        # Import data into Python as a PD Data Frame

        fullfilename = "C:\\Users\\fleboeuf\\Documents\\Programmation\\vicon-plugin\\Capture.U Practice Scripts\\Practice_Python\\Practice_GlobalAngles.csv"
        trial_new_deg, trial_helical = Vicon_practice_GlobalAngle(fullfilename)


        imuTranslators  = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER +"IMU\\","viconBlueTrident.translators")
       
        irp = imuReaderProcedures.CsvProcedure(fullfilename, imuTranslators["Translators"] )
        irf = imuFilters.ImuReaderFilter(irp)
        imu1 = irf.run()


        motProc = imuMotionProcedure.RealignedMotionProcedure()
        motFilter = imuFilters.ImuMotionFilter(imu1,motProc)
        motFilter.run()

        plot1 = plt.figure(1)
        plt.plot(trial_new_deg[:, 0], "-r", label="x")
        plt.plot(trial_new_deg[:, 1], "-g", label="y")
        plt.plot(trial_new_deg[:, 2], "-b", label="z")

        plt.plot(np.rad2deg(imu1.getAngleAxis()[:, 0]), "-*r", label="x2")
        plt.plot(np.rad2deg(imu1.getAngleAxis()[:, 1]), "-*g", label="y2")
        plt.plot(np.rad2deg(imu1.getAngleAxis()[:, 2]), "-*b", label="z2")

        plt.xlabel("Frames", loc='right')
        plt.ylabel("Angle (degrees)")
        plt.title("Re-aligned Orientation")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=5)
        plt.show()


    def test_Vicon_downsample(self):

        data_path = pyCGM2.TEST_DATA_PATH + "Opensense\\nexus\\"    

        imu_data = files.openPickleFile(data_path+"OpenSenseOutputs_fromVicon\\","imu_data")# np.load(data_path+"OpenSenseOutputs_fromVicon\\imu_data.npy",allow_pickle=True)
        imu_data_ds = files.openPickleFile(data_path+"OpenSenseOutputs_fromVicon\\","imu_data_ds")#np.load(data_path+"OpenSenseOutputs_fromVicon\\imu_data_ds.npy",allow_pickle=True)

        dynamicFile = "Walk.c3d"

        irp = imuReaderProcedures.C3dBlueTridentProcedure(data_path+dynamicFile,"1")
        irp.downsample(50.0)
        irf = imuFilters.ImuReaderFilter(irp)
        imu1 = irf.run()

        fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
        ax1.plot(np.rad2deg(imu1.getAngleAxis()[:,0]))
        ax1.plot(imu_data_ds["pelvis_imu"]["Global Angle"]["x"],"-or")
        
        ax2.plot(np.rad2deg(imu1.getAngleAxis()[:,1]))
        ax2.plot(imu_data_ds["pelvis_imu"]["Global Angle"]["y"],"-or")

        ax3.plot(np.rad2deg(imu1.getAngleAxis()[:,2]))
        ax3.plot(imu_data_ds["pelvis_imu"]["Global Angle"]["z"],"-or")
        plt.show()


