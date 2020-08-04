# coding: utf-8
# pytest -s --disable-pytest-warnings  test_analysis.py::Test_Btk::test_btkReaderWriter

from __future__ import unicode_literals
import pytest
import numpy as np

import pyCGM2
from pyCGM2.Lib import analysis

class Test_CGM1:
    def test_processing(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hånnibøl Lecter\\"

        modelledFilenames = ["gait Trial 01.c3d", "gait Trial 02.c3d"]

        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                            modelledFilenames,
                            type="Gait")

        assert analysisInstance.kinematicStats.data["LHipAngles","Left"]["values"].__len__() == 4
        assert analysisInstance.kinematicStats.data["RHipAngles","Right"]["values"].__len__() == 2

        assert analysisInstance.kineticStats.data["LHipMoment","Left"]["values"].__len__() == 2
        assert analysisInstance.kineticStats.data["RHipMoment","Right"]["values"].__len__() == 1
