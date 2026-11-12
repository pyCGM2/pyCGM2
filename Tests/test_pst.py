# coding: utf-8
# pytest -s --log-cli-level=INFO --disable-pytest-warnings  test_pst.py


import pytest
import numpy as np
from pyCGM2.Mek.mek import  mekCycle
import pyCGM2
from pyCGM2.Lib.Processing import spatioTemp

class Test_pst:
    def test_mekCycle(self):

            path = pyCGM2.TEST_DATA_PATH + "mek\\gaitdata\\"
            modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

            pstsData = spatioTemp.computeSpatioTemporalParameters(path, modelledFilenames)
