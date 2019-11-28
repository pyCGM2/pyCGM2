# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_forcePlate_lowLevel.py::Test_correctForcePlate::test_correctForcePlateType5
"""
Created on Thu Jul 07 15:14:18 2016

@author: aaa34169
"""
import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Utils import files


class Test_correctForcePlate():

    def test_correctForcePlateType5(self):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ForcePlate\\ForcePlateTypeManagement\\"
        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"LowLevel\\ForcePlate\\ForcePlateTypeManagement\\"
        files.createDir(DATA_PATH_OUT)


        btkAcq = btkTools.smartReader(MAIN_PATH + "HUG_gait_type5.c3d")

        forceplates.correctForcePlateType5(btkAcq)

        btkTools.smartWriter(btkAcq,DATA_PATH_OUT+ "HUG_gait_type5.c3d")
