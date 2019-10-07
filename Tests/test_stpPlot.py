# -*- coding: utf-8 -*-
# pytest --disable-pytest-warnings  test_stpPlot.py::Test_stpPlot::test_singleAnalysis

import logging

import matplotlib.pyplot as plt
import numpy as np

# pyCGM2 settings
import pyCGM2

from pyCGM2.Lib import analysis
from pyCGM2 import enums
from pyCGM2.Processing import c3dManager
from pyCGM2.Model.CGM2 import  cgm,cgm2

from pyCGM2.Report import plot,plotFilters,plotViewers,normativeDatasets
from pyCGM2.Tools import trialTools
from pyCGM2.Report import plot


class Test_stpPlot():

    def test_singleAnalysis(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot_latin1_çà\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]


        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisInstance = analysis.makeAnalysis(DATA_PATH,modelledFilenames)


        # viewer
        kv = plotViewers.SpatioTemporalPlotViewer(analysisInstance)
        kv.setNormativeDataset(normativeDatasets.NormalSTP())

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.setExport(DATA_PATH,"stp","pdf")
        pf.plot()

        plt.show()
