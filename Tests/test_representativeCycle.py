# coding: utf-8
#pytest -s --mpl --disable-pytest-warnings  test_representativeCycle.py::Test_lowLevel::test_Sangeux

import pytest


import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Lib import analysis
from pyCGM2.Processing import representative



def dataTest1():
    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hannibal Lecter\\"
    modelledFilenames = ["gait Trial 01.c3d", "gait Trial 02.c3d"]
    analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        modelledFilenames,
                        type="Gait")
    return DATA_PATH,analysisInstance





class Test_lowLevel:


    def test_Sangeux(self):

        DATA_PATH,analysisInstance = dataTest1()

        procedure = representative.Sangeux2015Procedure()
        procedure.setDefaultData()

        filt = representative.RepresentativeCycleFilter(analysisInstance, procedure)
        representativeIndex = filt.run()
        import ipdb; ipdb.set_trace()
