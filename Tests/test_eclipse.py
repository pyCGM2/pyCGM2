# coding: utf-8
# pytest -s --disable-pytest-warnings  test_eclipse.py::Test_eclipse::test_eclipse
from __future__ import unicode_literals

import pyCGM2
from pyCGM2 import enums
from pyCGM2.Eclipse import eclipse

class Test_eclipse:

    def test_eclipse(self):

        files = eclipse.getEnfFiles(pyCGM2.TEST_DATA_PATH+"LowLevel\\eclipse\\Hännibål\\", enums.EclipseType.Trial)


    def test_findCalibration(self):
        calib = eclipse.findCalibration(pyCGM2.TEST_DATA_PATH+"LowLevel\\eclipse\\Hännibål\\")
        assert calib == "PN01OP01S01STAT.Trial.enf"

    def test_findMotions(self):
        motions = eclipse.findMotions(pyCGM2.TEST_DATA_PATH+"LowLevel\\eclipse\\Hännibål\\")
        assert motions == ['PN01OP01S01SS02.Trial.enf', 'PN01OP01S01SS02[CGM1].Trial.enf', 'PN01OP01S01SS03[CGM1].Trial.enf']
