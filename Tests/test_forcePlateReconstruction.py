# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_forcePlateReconstruction.py

import pytest

import pandas as pd
import btk
import numpy as np
from streamlit import dataframe
from pyCGM2.Tools import btkTools
import pyCGM2

import matplotlib.pyplot as plt

from pyCGM2.Model.Models import singleBody

def display(acqui, seg,targetPointLabel):
    
    valX=np.zeros((acqui.GetPointFrameNumber(),3))
    valY=np.zeros((acqui.GetPointFrameNumber(),3))
    valZ=np.zeros((acqui.GetPointFrameNumber(),3))

    ref = seg.getReferential("TF")

    for i in range(0,acqui.GetPointFrameNumber()):
        valX[i,:]= np.dot(ref.motion[i].getRotation() , np.array([100.0,0.0,0.0])) + ref.motion[i].getTranslation()
        valY[i,:]= np.dot(ref.motion[i].getRotation() , np.array([0.0,100.0,0.0])) + ref.motion[i].getTranslation()
        valZ[i,:]= np.dot(ref.motion[i].getRotation() , np.array([0.0,0.0,100.0])) + ref.motion[i].getTranslation()

    btkTools.smartAppendPoint(acqui,targetPointLabel+"_X",valX,desc="")
    btkTools.smartAppendPoint(acqui,targetPointLabel+"_Y",valY,desc="")
    btkTools.smartAppendPoint(acqui,targetPointLabel+"_Z",valZ,desc="")
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

class Test_bioware:
    def test_reader(self):

     

        dataPath = pyCGM2.TEST_DATA_PATH + "LowLevel\\ForcePlate\\ForcePlateReconstructionFromFiles\\"
        # Data_FP = PFF_read("pdf_test_04.txt", 20)
        filename = "BASKET_001_01.txt"
        dataframe = pd.read_csv(dataPath+filename, sep="\t", skiprows=17)
       

        dataframe = dataframe.drop(dataframe.index[0]).reset_index(drop=True)
        dataframe["abs time (s)"] = dataframe["abs time (s)"] - dataframe["abs time (s)"].iloc[0]
        dataframe.rename(
            columns={"Fx": "Fx.0", "Fy": "Fy.0", "Fz": "Fz.0",
                     "Mx": "Mx.0", "My": "My.0", "Mz": "Mz.0",
                     "Ax": "Ax.0", "Ay": "Ay.0",
                     "Tz": "Tz.0",
                     "|Ft|": "|Ft|.0"},
            inplace=True
        )

        dataframe_100Hz = dataframe.iloc[::1]
        framerate = 1000
        pointFrameNumber = dataframe_100Hz.shape[0]-1
        analogFrameNumber = pointFrameNumber*1
        firstFrame = 1
        numberAnalogSamplePerFrame = 1


        forcePlateNumber = 6

        plateformes = {
            "corners": [
                np.array([[1198.238403, 4.822680, 14.086773, 1206.818970],
                        [25.089758, 14.248288, 9.559557, 20.146399],
                        [609.316406, 593.136108, 8.601675, 24.928083]]),
                np.array([[2401.281982, 1198.238403, 1206.818970, 2407.863770],
                        [41.392097, 25.089758, 20.146399, 34.714527],
                        [622.124756, 609.316406, 24.928083, 39.375587]]),
                np.array([[3593.567627, 2401.281982, 2407.863770, 3608.389404],
                        [52.807636, 41.392097, 34.714527, 47.214912],
                        [634.808289, 622.124756, 39.375587, 53.870922]]),
                np.array([[4791.291504, 3593.567627, 3608.389404, 4799.288574],
                        [67.429314, 52.807636, 47.214912, 59.639603],
                        [646.040222, 634.808289, 53.870922, 65.540871]]),
                np.array([[5398.644043, 4808.461426, 4819.291504, 5400.902344],
                        [74.150444, 68.518448, 52.499397, 61.907368],
                        [956.737488, 949.703979, -234.237350, -226.804794]]),
                np.array([[6598.154297, 5410.226562, 5417.708496, 6606.487305],
                        [85.979172, 70.353424, 64.846947, 79.658173],
                        [671.753479, 657.335876, 70.654427, 81.201256]])],
            }
        np.mean(plateformes["corners"][0], axis=1)
                
        # for i, arr in enumerate(plateformes["corners"]):
        #     plateformes["corners"][i] = arr[:, [0, 3, 2, 1]]

        # for i, arr in enumerate(plateformes["corners"]):
        #     plateformes["corners"][i] = arr[:, [3, 2, 1, 0]]

        plateformes["origins"] = [np.array([0.0,0,0]),
                                  np.array([0.0,0,0]),
                                  np.array([0.0,0,0]),
                                 np.array([0.0,0,-0]),
                                  np.array([0.0,0,0]),
                                  np.array([0.0,0,0])]
           

        acq = btk.btkAcquisition()
        acq.Init(0, int(pointFrameNumber), 0,
                            numberAnalogSamplePerFrame)
        acq.SetPointFrequency(framerate)
        acq.SetFirstFrame(firstFrame)

        rigidBodies = []
        for i_fp in range(forcePlateNumber):
            btkTools.smartAppendPoint(acq, f"ForcePlate{i_fp}_p0",plateformes["corners"][i_fp][:,0]* np.ones((acq.GetPointFrameNumber(),3)))
            btkTools.smartAppendPoint(acq, f"ForcePlate{i_fp}_p1",plateformes["corners"][i_fp][:,1]* np.ones((acq.GetPointFrameNumber(),3)))
            btkTools.smartAppendPoint(acq, f"ForcePlate{i_fp}_p2",plateformes["corners"][i_fp][:,2]* np.ones((acq.GetPointFrameNumber(),3)))
            btkTools.smartAppendPoint(acq, f"ForcePlate{i_fp}_p3",plateformes["corners"][i_fp][:,3]* np.ones((acq.GetPointFrameNumber(),3)))
            btkTools.smartAppendPoint(acq, f"ForcePlate{i_fp}_Origin", plateformes["origins"][i_fp]* np.ones((acq.GetPointFrameNumber(),3)))    


            rigidBody = singleBody.SingleBody([f"ForcePlate{i_fp}_p2",f"ForcePlate{i_fp}_p3",f"ForcePlate{i_fp}_p0",f"ForcePlate{i_fp}_Origin"],
                                        [f"ForcePlate{i_fp}_p2",f"ForcePlate{i_fp}_p3",f"ForcePlate{i_fp}_p0"],sequence=  "YZX")
            rigidBody.calibrate(acq) #,calibFramesOfInterest=[66,66])
            rigidBody.fit(acq)
            rigidBodies.append(rigidBody)
        
        rigidBodies[0].displayAxis(acq,"ForcePlate0_p0")


        # i_fp = 0    
        # forcesValues = dataframe[[f"Fx.{i_fp}",f"Fy.{i_fp}",f"Fz.{i_fp}"]].to_numpy().astype(np.float64)[:analogFrameNumber]
        # globalForces = rigidBody.globalize(-forcesValues)



        for i_fp in range(forcePlateNumber):


            forcesValues = dataframe[[f"Fx.{i_fp}",f"Fy.{i_fp}",f"Fz.{i_fp}"]].to_numpy().astype(np.float64)[:analogFrameNumber]
            momentValues = dataframe[[f"Mx.{i_fp}",f"My.{i_fp}",f"Mz.{i_fp}"]].to_numpy().astype(np.float64)[:analogFrameNumber]
            positionValues = dataframe[[f"Ax.{i_fp}",f"Ay.{i_fp}"]].to_numpy().astype(np.float64)[:analogFrameNumber]

            globalForces = rigidBodies[i_fp].globalize(-forcesValues)
            globalMoments = rigidBodies[i_fp].globalize(-momentValues)

            # t = np.arange(forcesValues.shape[0])
            # plt.plot(t, forcesValues[:, 0], label="Fx")
            # plt.plot(t, forcesValues[:, 1], label="Fy")
            # plt.plot(t, forcesValues[:, 2], label="Fz")
            # plt.legend()
            # plt.show()

            positionValues = np.hstack((positionValues, np.zeros((positionValues.shape[0], 1))))

            # a iterer ici
            forceLabels = [
                "Force.Fx"+str(i_fp+1), "Force.Fy"+str(i_fp+1), "Force.Fz"+str(i_fp+1)]
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
                "Moment.Mx"+str(i_fp+1), "Moment.My"+str(i_fp+1), "Moment.Mz"+str(i_fp+1)]
            for j in range(0, 3):
                analog = btk.btkAnalog()
                analog.SetLabel(momentLabels[j])
                analog.SetUnit("Nmm")  
                analog.SetFrameNumber(analogFrameNumber)
                analog.SetValues(-1000*momentValues[:,j])
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

        origins = [] 
        for i_fp in range(forcePlateNumber):
            platform_center = plateformes["origins"][i_fp]
            origins.append(platform_center)
        # fin de partie a iterer

        md_origin = btk.btkMetaData('ORIGIN')
        md_origin.SetInfo(btk.btkMetaDataInfo(
            [3, int(forcePlateNumber)], np.concatenate(origins)))
        md_force_platform.AppendChild(md_origin)
        

        # a iterer ici
        corners = [] 
        for i_fp in range(forcePlateNumber):
            corners.append(plateformes["corners"][i_fp].T.flatten())
        # fin de partie a iterer

        md_corners = btk.btkMetaData('CORNERS')
        md_corners.SetInfo(btk.btkMetaDataInfo(
            [3, 4, int(forcePlateNumber)], np.concatenate(corners)))
        md_force_platform.AppendChild(md_corners)

        md_channel = btk.btkMetaData('CHANNEL')
        md_channel.SetInfo(btk.btkMetaDataInfo(
            [6, int(forcePlateNumber)], np.arange(1, int(forcePlateNumber)*6+1).tolist()))
        md_force_platform.AppendChild(md_channel)




        

       







        btkTools.smartWriter(acq,"testBiowareForcePlate.c3d")



