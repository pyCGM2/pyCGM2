# coding: utf-8
#pytest -s --mpl --disable-pytest-warnings  test_excelExport.py

import pytest



import pyCGM2
from pyCGM2.Lib import analysis


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
                        type="Gait",
                        emgChannels = None)
    return DATA_PATH,analysisInstance


def dataTest2():
    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 1 - CGM1\\"
    modelledFilenames = ["20180706_CS_PONC_S_NNNN dyn 02.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 03.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 05.c3d"]

    analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        modelledFilenames,
                        type="Gait",
                        emgChannels = emgChannels,
                        pointLabelSuffix=None,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        )


    return DATA_PATH,modelledFilenames,analysisInstance


def dataTest3():
    DATA_PATH1 = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 1 - CGM1\\"
    modelledFilenames1 = ["20180706_CS_PONC_S_NNNN dyn 02.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 03.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 05.c3d"]

    analysisInstance1 = analysis.makeAnalysis(DATA_PATH1,
                        modelledFilenames1,
                        type="Gait",
                        emgChannels = emgChannels,
                        pointLabelSuffix=None,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        )

    DATA_PATH2 = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 2 - CGM23\\"
    modelledFilenames2 = ["20200729-SC-PONC-S-NNNN dyn 04.c3d",
                        "20200729-SC-PONC-S-NNNN dyn 06.c3d"]

    analysisInstance2 = analysis.makeAnalysis(DATA_PATH2,
                        modelledFilenames2,
                        type="Gait",
                        emgChannels = emgChannels,
                        pointLabelSuffix=None,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        )


    return DATA_PATH1,modelledFilenames1,analysisInstance1,DATA_PATH2,modelledFilenames2,analysisInstance2



class Test_excelExport:


    def test_advanced(self):

        DATA_PATH,analysisInstance = dataTest1()
        print(DATA_PATH)
        analysis.exportAnalysis(analysisInstance,DATA_PATH,"dataset1_export")


        # DATA_PATH,modelledFilenames,analysisInstance = dataTest2()
        # print(DATA_PATH)
        # analysis.exportAnalysis(analysisInstance,DATA_PATH,"dataset2_export")

