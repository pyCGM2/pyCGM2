# coding: utf-8
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_BlueTrident::test_reader_csv
#pytest -s --disable-pytest-warnings  test_BlueTrident.py::Test_Garches::test_reader
# from __future__ import unicode_literals

import pandas as pd
import numpy as np

import pyCGM2
LOGGER = pyCGM2.LOGGER

from pyCGM2.Tools import btkTools
from pyCGM2.IMU.BlueTrident import BlueTrident
from pyCGM2.IMU import imu
from pyCGM2.Utils import files

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

        import ipdb; ipdb.set_trace()

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

        import ipdb; ipdb.set_trace()
