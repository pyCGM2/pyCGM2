# coding: utf-8
# pytest -s --disable-pytest-warnings  test_events.py::Test_gaitEvents::test_zeni



import logging
import matplotlib.pyplot as plt

import pyCGM2; LOGGER = pyCGM2.LOGGER

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Events import eventFilters
from pyCGM2.Events import eventProcedures

class Test_gaitEvents:
    def test_zeni(self):


        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\events\\gaitEvents\\"


        gaitFilename="gait Trial 01.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        evp = eventProcedures.ZeniProcedure()

        evf = eventFilters.EventFilter(evp,acq)
        evf.detect()

        #btkTools.smartWriter("testEvent0.c3d", acq)
