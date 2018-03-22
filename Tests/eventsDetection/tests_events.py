# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# btk
pyCGM2.CONFIG.addBtk()


# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Events import events



class test_Zeni():

    @classmethod
    def detection(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\event detection\\events\\"

        # --- Motion 1
        gaitFilename="gait-noEvents.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        evp = events.ZeniProcedure()

        evf = events.EventFilter(evp,acq)
        evf.detect()

        btkTools.smartWriter("testEvent0.c3d", acq)


if __name__ == "__main__":
    plt.close("all")

    test_Zeni.detection()
