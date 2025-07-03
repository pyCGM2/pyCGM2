# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_forcePlateReconstruction.py

import pytest

import pandas as pd
import btk
import numpy as np
from pyCGM2.Tools import btkTools
import pyCGM2

class Test_bertec:
    def test_onePF_lowLevel(self):
        
        dataPath = DATA_PATH =  pyCGM2.TEST_DATA_PATH + "LowLevel\\ForcePlate\\ForcePlateReconstruction\\"
        filename = "bertec-yannisDataSample.csv"
        dataframe =   pd.read_csv(dataPath+filename,sep=",",skiprows=4)

        dataframe["Time (s)"] = dataframe["Time (s)"] - dataframe["Time (s)"].iloc[0]
        dataframe_100Hz = dataframe.iloc[::10]

        framerate = 100
        pointFrameNumber = dataframe_100Hz.shape[0]-1
        analogFrameNumber = pointFrameNumber*10
        firstFrame = 1
        numberAnalogSamplePerFrame = 10


        forcePlateNumber = 1
        leftIndex = "1"

        forcesValues = dataframe[[leftIndex+":FX",leftIndex+":FY",leftIndex+":FZ"]].to_numpy()[:analogFrameNumber]
        momentValues = dataframe[[leftIndex+":MX",leftIndex+":MY",leftIndex+":MZ"]].to_numpy()[:analogFrameNumber]
        positionValues = dataframe[[leftIndex+":COPX",leftIndex+":COPY"]].to_numpy()[:analogFrameNumber]
        positionValues = np.hstack((positionValues, np.zeros((positionValues.shape[0], 1))))




        acq = btk.btkAcquisition()
        acq.Init(0, int(pointFrameNumber), 0,
                            numberAnalogSamplePerFrame)
        acq.SetPointFrequency(framerate)
        acq.SetFirstFrame(firstFrame)

        fp_count= 0

        # a iterer ici
        forceLabels = [
            "Force.Fx"+str(fp_count+1), "Force.Fy"+str(fp_count+1), "Force.Fz"+str(fp_count+1)]
        for j in range(0, 3):
            analog = btk.btkAnalog()
            analog.SetLabel(forceLabels[j])
            analog.SetUnit("N")  
            analog.SetFrameNumber(analogFrameNumber)
            analog.SetValues(-forcesValues[:, j])
            analog.SetDescription("test")
            #analog.SetGain(btk.btkAnalog.PlusMinus10)

            acq.AppendAnalog(analog)

        momentLabels = [
            "Moment.Mx"+str(fp_count+1), "Moment.My"+str(fp_count+1), "Moment.Mz"+str(fp_count+1)]
        for j in range(0, 3):
            analog = btk.btkAnalog()
            analog.SetLabel(momentLabels[j])
            analog.SetUnit("Nmm")  
            analog.SetFrameNumber(analogFrameNumber)
            analog.SetValues(-1000.0*momentValues[:,j])
            analog.SetDescription("test")
            #analog.GetGain(btk.btkAnalog.PlusMinus10)
            acq.AppendAnalog(analog)
        # fin de partie a iterer

        # metadata for platform type2
        md_force_platform = btk.btkMetaData('FORCE_PLATFORM')
        btk.btkMetaDataCreateChild(
            md_force_platform, "USED", int(forcePlateNumber))  # a
        btk.btkMetaDataCreateChild(md_force_platform, "ZERO", [1, 0])
        btk.btkMetaDataCreateChild(md_force_platform, "TYPE", btk.btkDoubleArray(
            forcePlateNumber, 2))  # btk.btkDoubleArray(forcePlateNumber, 2))# add a child

        acq.GetMetaData().AppendChild(md_force_platform)

        # a iterer ici
        origins = [] 
        origins.append(np.array([0.,0.,0.]))
        # fin de partie a iterer

        md_origin = btk.btkMetaData('ORIGIN')
        md_origin.SetInfo(btk.btkMetaDataInfo(
            [3, int(forcePlateNumber)], np.concatenate(origins)))
        md_force_platform.AppendChild(md_origin)
        

        # a iterer ici
        corners = [] 
        values = np.array([ 4.00000000e+02,  2.66823008e-10,  0.00000000e+00,  4.00234512e-10,
                            -2.66823008e-10,  0.00000000e+00, -4.00234512e-10,  6.00000000e+02,
                            0.00000000e+00,  4.00000000e+02,  6.00000000e+02,  0.00000000e+00])
        corners.append(values.T.flatten())
        # fin de partie a iterer

        md_corners = btk.btkMetaData('CORNERS')
        md_corners.SetInfo(btk.btkMetaDataInfo(
            [3, 4, int(forcePlateNumber)], np.concatenate(corners)))
        md_force_platform.AppendChild(md_corners)

        md_channel = btk.btkMetaData('CHANNEL')
        md_channel.SetInfo(btk.btkMetaDataInfo(
            [6, int(forcePlateNumber)], np.arange(1, int(forcePlateNumber)*6+1).tolist()))
        md_force_platform.AppendChild(md_channel)


        btkTools.smartWriter(acq,"testForcePlate2.c3d")




