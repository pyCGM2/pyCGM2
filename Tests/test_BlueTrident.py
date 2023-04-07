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
from pyCGM2.IMU.BlueTrident import BlueTridentReader
from pyCGM2.IMU import imu
from pyCGM2.Utils import files

from pyCGM2.IMU import imuFilters
from pyCGM2.IMU.Procedures import relativeImuAngleProcedures

class Test_BlueTrident:

    def test_reader(self):
        fullfilename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\BlueTrident\\trial 1.c3d"
        acq = btkTools.smartReader(fullfilename)

        imu1 = BlueTridentReader.btkGetBlueTrident(acq,"1")
        imu1.downsample()
        imu1.constructDataFrame()

        imu2 = BlueTridentReader.btkGetBlueTrident(acq,"2")
        imu2.downsample()
        imu2.constructDataFrame()

        imu3 = BlueTridentReader.btkGetBlueTrident(acq,"3")
        imu3.downsample()
        imu3.constructDataFrame()




    def test_reader_csv(self):
        fullfilename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\example_TS-01436.csv"

        imu1 = BlueTridentReader.readBlueTridentCsv(fullfilename,1125)
        imu1.downsample()
        imu1.constructDataFrame()


    def test_reader_multipleCsv(self):
        fullfilename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\Rouling Maxence_TS-02374_2022-04-26-16-34-56_lowg.csv"
        fullfilename1 = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\Rouling Maxence_TS-02122_2022-04-26-16-34-56_lowg.csv"
        fullfilename2 = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\\BlueTridentCaptureU\\Rouling Maxence_TS-01436_2022-04-26-16-34-56_lowg.csv"


        fullfilenames = [fullfilename,fullfilename1, fullfilename2]
        imus = BlueTridentReader.readmultipleBlueTridentCsv(fullfilenames,1125)

    def test_reader_ktk(self):
        fullfilename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\BlueTrident\\trial 1.c3d"
        acq = btkTools.smartReader(fullfilename)

        imu1 = BlueTridentReader.btkGetBlueTrident(acq,"1")
        imu1.downsample()
        imu1.constructTimeseries()

        imu2 = BlueTridentReader.btkGetBlueTrident(acq,"2")
        imu2.downsample()
        imu2.constructTimeseries()

        imu3 = BlueTridentReader.btkGetBlueTrident(acq,"3")
        imu3.downsample()
        imu3.constructTimeseries()


class Test_BlueTridentOrientation:

    def test_absoluteAngles(self):

        fullfilename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IMU\\BlueTrident-markers\\pycgm2-data01.c3d"
        acq = btkTools.smartReader(fullfilename)

        # ---- pycgm2
        imu1 = BlueTridentReader.btkGetBlueTrident(acq,"8") # get directly the viconID
        imu1.constructDataFrame()
        imu1.computeOrientations()
        imu1.computeAbsoluteAngles()

        plt.figure()
        plt.plot(imu1.m_absoluteAngles["eulerXYZ"][:,0])
        plt.show()

    def test_alignement(self):

        # fullfilename = pyCGM2.TEST_DATA_PATH + "IMU\\practice\\upperLimb\\upperLimb01.c3d"
        fullfilename = pyCGM2.TEST_DATA_PATH + "IMU\\angleMeasurement\\goniometer\\right36 -0to120 trial 01.c3d"


        acq = btkTools.smartReader(fullfilename)

        # ---- pycgm2
        imu1 = BlueTridentReader.btkGetBlueTrident(acq,"1") # get directly the viconID
        imu1.constructDataFrame()
        imu1.computeOrientations()


        # ---- pycgm2
        imu2 = BlueTridentReader.btkGetBlueTrident(acq,"2") # get directly the viconID
        imu2.constructDataFrame()
        imu2.computeOrientations()


        proc = relativeImuAngleProcedures.BlueTridentsRelativeAnglesProcedure(representation = "Euler", eulerSequence="ZYX")
        filt =  imuFilters.RelativeIMUAnglesFilter(imu1,imu2, proc)
        jointFinalValues = filt.compute()

        plt.figure()
        plt.plot(jointFinalValues[:,0],"-r")
        plt.plot(jointFinalValues[:,1],"-g")
        plt.plot(jointFinalValues[:,2],"-b")


        imu2.align(imu1)

        proc = relativeImuAngleProcedures.BlueTridentsRelativeAnglesProcedure(representation = "Euler", eulerSequence="ZYX")
        filt =  imuFilters.RelativeIMUAnglesFilter(imu1,imu2, proc)
        jointFinalValues = filt.compute()

        plt.figure()
        plt.plot(jointFinalValues[:,0],"-r")
        plt.plot(jointFinalValues[:,1],"-g")
        plt.plot(jointFinalValues[:,2],"-b")


        plt.show()
         

    def test_relativeAngles(self):

        # fullfilename = pyCGM2.TEST_DATA_PATH + "IMU\\practice\\upperLimb\\upperLimb01.c3d"
        fullfilename = pyCGM2.TEST_DATA_PATH + "IMU\\angleMeasurement\\goniometer\\right36 -0to120 trial 01.c3d"


        acq = btkTools.smartReader(fullfilename)

        # ---- pycgm2
        imu1 = BlueTridentReader.btkGetBlueTrident(acq,"1") # get directly the viconID
        imu1.constructDataFrame()
        imu1.computeOrientations()


        # ---- pycgm2
        imu2 = BlueTridentReader.btkGetBlueTrident(acq,"2") # get directly the viconID
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
        
        accel = np.array( [data1["acc_x"], data1["acc_y"], data1["acc_z"]]).T
        angularVelocity = np.array( [data1["gyro_x"], data1["gyro_y"], data1["gyro_z"]]).T

        imu1 = imu.Imu(128, accel,angularVelocity, mag=None)
        imu1.constructDataFrame()

