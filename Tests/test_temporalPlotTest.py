# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# pytest --disable-pytest-warnings  test_temporalPlotTest.py::Test_PlotTest::test_temporalKinematicPlotPanel
import ipdb
import logging


import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
from pyCGM2.Report import plot,plotFilters,plotViewers
from pyCGM2.Tools import trialTools

class Test_PlotTest():

    def test_temporalKinematicPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]


        trial =trialTools.smartTrialReader(DATA_PATH,modelledFilenames[0])

        # # viewer
        kv = plotViewers.TemporalKinematicsPlotViewer(trial)
        #
        # # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        #plt.show()


    def test_temporalKineticPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        trial =trialTools.smartTrialReader(DATA_PATH,modelledFilenames[0])
        # viewer
        kv = plotViewers.TemporalKineticsPlotViewer(trial)

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        #plt.show()
