# coding: utf-8
#pytest -s --mpl --disable-pytest-warnings  Test_c3dmanager.py

# from __future__ import unicode_literals
import pytest


import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Lib import analysis
from pyCGM2.Utils import files




class Test_c3dmanager:

    #@pytest.mark.mpl_image_compare
    def test_1(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hannibal Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d", "gait Trial 02.c3d"]
        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                            modelledFilenames,
                            type="Gait")
