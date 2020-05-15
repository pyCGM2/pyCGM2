# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# pytest -s --disable-pytest-warnings  test_highLevelPlot.py::Test_emgPlotTests::test_temporalEmgPlot
import logging
import os
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Configurator import EmgManager
from pyCGM2.Utils import files
from pyCGM2.Tools import btkTools, trialTools
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot

class Test_emgPlotTests():

    def test_temporalEmgPlot(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\\EMG\\Hånnibøl Lecter-nerve block\\"
        gaitTrial = "PRE-gait trial 01.c3d"
        restTrial = "PRE-repos.c3d"

        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"GaitData\\EMG\\Hånnibøl Lecter-nerve block\\"
        files.createDir(DATA_PATH_OUT)

        #--------------------------settings-------------------------------------
        if os.path.isfile(DATA_PATH + "emg.settings"):
            emgSettings = files.openFile(DATA_PATH,"emg.settings")
            logging.warning("[pyCGM2]: emg.settings detected in the data folder")
        else:
            emgSettings = None


        manager = EmgManager.EmgConfigManager(None,localInternalSettings=emgSettings)
        manager.contruct()

        EMG_LABELS,EMG_MUSCLES,EMG_CONTEXT,NORMAL_ACTIVITIES  =  manager.getEmgConfiguration()

        rectifyBool=True

        acq = btkTools.smartReader(DATA_PATH+gaitTrial)


        analysis.processEMG_fromBtkAcq(acq, EMG_LABELS,
            highPassFrequencies=[20,200],
            envelopFrequency=6.0)

        openmaTrial = trialTools.convertBtkAcquisition(acq)

        plot.plotTemporalEMG(DATA_PATH,gaitTrial, EMG_LABELS,EMG_MUSCLES, EMG_CONTEXT, NORMAL_ACTIVITIES,exportPdf=False,rectify=rectifyBool,openmaTrial=openmaTrial)
