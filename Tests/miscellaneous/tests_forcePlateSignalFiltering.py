# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 15:14:18 2016

@author: aaa34169
"""

import logging
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2 import btk


from pyCGM2.Tools import  btkTools
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Signal import signal_processing

import numpy as np



class test_ForcePlateSignalFiltering():

    @classmethod
    def saveInC3d(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1
        gaitFilename="walking_oppositeX_2pf.c3d"
        acqGait0 = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        acqGait = btk.btkAcquisition.Clone(acqGait0)
        signal_processing.forcePlateFiltering(acqGait)


        pfe0 = btk.btkForcePlatformsExtractor()
        pfe0.SetInput(acqGait0)
        pfc0 = pfe0.GetOutput()
        pfc0.Update()

        pfe1 = btk.btkForcePlatformsExtractor()
        pfe1.SetInput(acqGait)
        pfc1 = pfe1.GetOutput()
        pfc1.Update()



        plt.figure()
        plt.plot(acqGait0.GetAnalog("Force.Fx1").GetValues()[:,0],"-r")
        plt.plot(acqGait.GetAnalog("Force.Fx1").GetValues()[:,0],"ob")

        plt.figure()
        plt.plot(pfc0.GetItem(0).GetChannel(0).GetValues()[:,0],"-r")
        plt.plot(pfc1.GetItem(0).GetChannel(0).GetValues()[:,0],"-b")
        plt.show()


if __name__ == "__main__":
    plt.close("all")

    test_ForcePlateSignalFiltering.saveInC3d()
