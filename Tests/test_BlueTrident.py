# coding: utf-8
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_BlueTrident::test_reader_csv
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_BlueTridentOrientation::test_globalAngles
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_Garches::test_reader
# from __future__ import unicode_literals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pyCGM2
LOGGER = pyCGM2.LOGGER

from pyCGM2.Tools import btkTools
from pyCGM2.IMU.BlueTrident import BlueTrident
from pyCGM2.IMU import imu
from pyCGM2.Utils import files

from pyCGM2.IMU import imuFilters
from pyCGM2.IMU.Procedures import relativeImuAngleProcedures

class Test_BlueTrident:

    def test_reader(self):
        fullfilename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\BlueTrident\\trial 1.c3d"
        acq = btkTools.smartReader(fullfilename)

        acq = BlueTrident.correctBlueTridentIds(acq)

        imu1 = BlueTrident.getBlueTrident(acq,"1")
        imu1.downsample()
        imu1.constructDataFrame()

        imu2 = BlueTrident.getBlueTrident(acq,"2")
        imu2.downsample()
        imu2.constructDataFrame()

        imu3 = BlueTrident.getBlueTrident(acq,"3")
        imu3.downsample()
        imu3.constructDataFrame()




    def test_reader_csv(self):
        fullfilename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\example_TS-01436.csv"

        imu1 = BlueTrident.readBlueTridentCsv(fullfilename,1125)
        imu1.downsample()
        imu1.constructDataFrame()


    def test_reader_multipleCsv(self):
        fullfilename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\Rouling Maxence_TS-02374_2022-04-26-16-34-56_lowg.csv"
        fullfilename1 = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\Rouling Maxence_TS-02122_2022-04-26-16-34-56_lowg.csv"
        fullfilename2 = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\Rouling Maxence_TS-01436_2022-04-26-16-34-56_lowg.csv"


        fullfilenames = [fullfilename,fullfilename1, fullfilename2]
        imus = BlueTrident.readmultipleBlueTridentCsv(fullfilenames,1125)

    def test_reader_ktk(self):
        fullfilename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\BlueTrident\\trial 1.c3d"
        acq = btkTools.smartReader(fullfilename)
        acq = BlueTrident.correctBlueTridentIds(acq)

        imu1 = BlueTrident.getBlueTrident(acq,"1")
        imu1.downsample()
        imu1.constructTimeseries()

        imu2 = BlueTrident.getBlueTrident(acq,"2")
        imu2.downsample()
        imu2.constructTimeseries()

        imu3 = BlueTrident.getBlueTrident(acq,"3")
        imu3.downsample()
        imu3.constructTimeseries()


class Test_BlueTridentOrientation:

    def test_globalAngles(self):

        fullfilename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\BlueTrident-markers\\pycgm2-data01.c3d"
        acq = btkTools.smartReader(fullfilename)

        # ---- pycgm2
        imu1 = BlueTrident.getBlueTrident(acq,"8") # get directly the viconID
        imu1.constructDataFrame()
        imu1.computeOrientations()

        plt.figure()
        plt.plot(imu1.m_data["Orientations"]["ViconGlobalAngles"]["eulerXYZ"][:,0])
        plt.show()

    def test_relativeAngles(self):

        # fullfilename = pyCGM2.TEST_DATA_PATH + "IMU\\practice\\upperLimb\\upperLimb01.c3d"
        fullfilename = pyCGM2.TEST_DATA_PATH + "IMU\\angleMeasurement\\goniometer\\right36 -0to120 trial 01.c3d"


        acq = btkTools.smartReader(fullfilename)

        # ---- pycgm2
        imu1 = BlueTrident.getBlueTrident(acq,"1") # get directly the viconID
        imu1.constructDataFrame()
        imu1.computeOrientations()


        # ---- pycgm2
        imu2 = BlueTrident.getBlueTrident(acq,"2") # get directly the viconID
        imu2.constructDataFrame()
        imu2.computeOrientations()


        # return the angle rotation from the global axis
        proc = relativeImuAngleProcedures.BlueTridentsRelativeAnglesProcedure(representation = "GlobalAngle")
        filt =  imuFilters.RelativeIMUAnglesFilter(imu1,imu2, proc)
        jointFinalValues = filt.compute()

        plt.figure()
        plt.plot(jointFinalValues[:,0],"-r")
        plt.plot(jointFinalValues[:,1],"-g")
        plt.plot(jointFinalValues[:,2],"-b")

        # return the angle from euler sequence
        proc = relativeImuAngleProcedures.BlueTridentsRelativeAnglesProcedure(representation = "Euler", eulerSequence="ZYX")
        filt =  imuFilters.RelativeIMUAnglesFilter(imu1,imu2, proc)
        jointFinalValues = filt.compute()

        plt.figure()
        plt.plot(jointFinalValues[:,0],"-r")
        plt.plot(jointFinalValues[:,1],"-g")
        plt.plot(jointFinalValues[:,2],"-b")


        plt.show()
         






class Test_Garches:

    def test_reader(self):

        data1 = files.openFile( pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\GarcheIMU\\","E1_Roue Droite_raw.json")
        imu1 = imu.Imu(128)
        imu1.setAcceleration("X", np.array(data1["acc_x"]))
        imu1.setAcceleration("Y", np.array(data1["acc_y"]))
        imu1.setAcceleration("Z", np.array(data1["acc_z"]))

        imu1.setGyro("X", np.array(data1["gyro_x"]))
        imu1.setGyro("Y", np.array(data1["gyro_y"]))
        imu1.setGyro("Z", np.array(data1["gyro_z"]))

        imu1.constructDataFrame()

