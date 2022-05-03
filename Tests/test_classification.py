# coding: utf-8
#pytest -s --mpl --disable-pytest-warnings  test_classification.py::Test_lowLevel::test_PFKE

import pytest
from pyCGM2.Report import normativeDatasets

import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Lib import analysis
from pyCGM2.Processing import classification
from pyCGM2.Utils import files


def dataTest1():
    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hannibal Lecter\\"
    modelledFilenames = ["gait Trial 01.c3d", "gait Trial 02.c3d"]
    analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        modelledFilenames,
                        type="Gait")
    return DATA_PATH,analysisInstance





class Test_lowLevel:


    def test_PFKE(self):

        DATA_PATH,analysisInstance = dataTest1()

        nds = normativeDatasets.NormativeData("Schwartz2008","Free")

        procedure = classification.PFKEprocedure(nds)

        filt = classification.ClassificationFilter(analysisInstance, procedure)
        sagClass = filt.run()

        procedure.plot(analysisInstance)

    def test_PFKE2(self):

        analysisInstance = files.loadAnalysis("C:\\Users\\fleboeuf\\Documents\\ANALYSES\\inclined walk\\2021-july-reprocessing\\results\\1IC-CP\\Processing-CGM21\\", "Condition1")


        nds = normativeDatasets.NormativeData("Schwartz2008","Free")
        procedure = classification.PFKEprocedure(nds)
        filt = classification.ClassificationFilter(analysisInstance, procedure)
        sagClass = filt.run()

        procedure.plot(analysisInstance)
        print (sagClass)
        plt.show()
