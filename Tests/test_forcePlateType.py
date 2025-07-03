# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_forcePlateType.py::Test_forcePlateTypes::test_forcePlateBertec3digital
import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.ForcePlates import forceplates

import pyCGM2; LOGGER = pyCGM2.LOGGER

from pyCGM2.Lib.CGM import  cgm1
from pyCGM2.Model import modelFilters
from pyCGM2.Nexus import vskTools
from pyCGM2.Utils import testingUtils


class Test_forcePlateTypes():

    def test_forcePlateType4(self):

        DATA_PATH =  pyCGM2.TEST_DATA_PATH + "LowLevel\\ForcePlate\\ForcePlateTypeManagement\\"
        acq = btkTools.smartReader(DATA_PATH+"Qualisys-bertec ForcePlate4.c3d")
        btkTools.smartWriter(acq,"test.c3d")

    def test_forcePlateBertec3digital(self):

        DATA_PATH =  pyCGM2.TEST_DATA_PATH + "Issues\\issue_digitalBertec_3FP\\Bertec data\\"
        acq = btkTools.smartReader(DATA_PATH+"Gait LB - CGM2 2-fromQTM.c3d")
        btkTools.smartWriter(acq,DATA_PATH+"test.c3d")

