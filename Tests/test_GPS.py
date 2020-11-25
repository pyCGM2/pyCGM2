# coding: utf-8
#pytest -s --disable-pytest-warnings  test_GPS.py::Test_GPS::test_CGM1
# from __future__ import unicode_literals

import pyCGM2
from pyCGM2.Lib import analysis
from pyCGM2.Processing import scores

from pyCGM2.Report import normativeDatasets


normativeDataset = normativeDatasets.Schwartz2008("Free")
newNormativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

emgChannels=['Voltage.EMG1','Voltage.EMG2','Voltage.EMG3','Voltage.EMG4','Voltage.EMG5',
            'Voltage.EMG6','Voltage.EMG7','Voltage.EMG8','Voltage.EMG9','Voltage.EMG10']

muscles=['RF','RF','VL','VL','HAM',
            'HAM','TI','TI','SOL','SOL']

contexts=['Left','Right','Left','Right','Left',
            'Right','Left','Right','Left','Right']

normalActivityEmgs=['RECFEM','RECFEM', None,None,None,
            None,None,None,None,None]


def dataTest1():
    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hannibal Lecter\\"
    modelledFilenames = ["gait Trial 01.c3d", "gait Trial 02.c3d"]
    analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        modelledFilenames,
                        type="Gait")
    return DATA_PATH,analysisInstance


def dataTest2():
    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 1 - CGM1\\"
    modelledFilenames = ["20180706_CS_PONC_S_NNNN dyn 02.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 03.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 05.c3d"]

    emgFilenames = modelledFilenames


    analysisInstance = analysis.makeCGMGaitAnalysis(DATA_PATH,modelledFilenames,emgFilenames,emgChannels,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        pointLabelSuffix=None)


    return DATA_PATH,modelledFilenames,analysisInstance


def dataTest3():
    DATA_PATH1 = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 1 - CGM1\\"
    modelledFilenames1 = ["20180706_CS_PONC_S_NNNN dyn 02.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 03.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 05.c3d"]

    emgFilenames1 = modelledFilenames1

    analysisInstance1 = analysis.makeCGMGaitAnalysis(DATA_PATH1,modelledFilenames1,emgFilenames1,emgChannels,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        pointLabelSuffix=None)

    DATA_PATH2 = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 2 - CGM23\\"
    modelledFilenames2 = ["20200729-SC-PONC-S-NNNN dyn 04.c3d",
                        "20200729-SC-PONC-S-NNNN dyn 06.c3d"]

    emgFilenames2 = modelledFilenames2

    analysisInstance2 = analysis.makeCGMGaitAnalysis(DATA_PATH2,modelledFilenames2,emgFilenames2,emgChannels,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        pointLabelSuffix=None)


    return DATA_PATH1,modelledFilenames1,analysisInstance1,DATA_PATH2,modelledFilenames2,analysisInstance2



class Test_GPS:

    def test_CGM1(self):

       #  results = {'Overall': {'std': array([0.2023908]), 'values': array([5.84457972, 5.35628479, 5.98072926, 5.92348919, 5.73983361,
       # 5.75431847]), 'median': array([5.79944909]), 'mean': array([5.76653917])}, 'Context': {'Right': {'std': array([0.24414746]), 'values': array([5.84457972, 5.35628479]), 'median': array([5.60043226]), 'mean': array([5.60043226])}, 'Left': {'std': array([0.10462042]), 'values': array([5.98072926, 5.92348919, 5.73983361, 5.75431847]), 'median': array([5.83890383]), 'mean': array([5.84959263])}}}

        DATA_PATH,analysisInstance = dataTest1()

        gps =scores.CGM1_GPS()
        scf = scores.ScoreFilter(gps,analysisInstance, newNormativeDataset)
        scf.compute()
