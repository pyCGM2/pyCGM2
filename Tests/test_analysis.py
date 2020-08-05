# coding: utf-8
# pytest -s --disable-pytest-warnings  test_analysis.py::Test_Btk::test_btkReaderWriter

from __future__ import unicode_literals
import pytest
import numpy as np
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.Lib import analysis

from pyCGM2.Report import plot


class Test_CGM1:
    def test_processing(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hånnibøl Lecter\\"

        modelledFilenames = ["gait Trial 01.c3d", "gait Trial 02.c3d"]

        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                            modelledFilenames,
                            type="Gait")


        assert analysisInstance.kinematicStats.data["LHipAngles","Left"]["values"].__len__() == 4
        assert analysisInstance.kinematicStats.data["RHipAngles","Right"]["values"].__len__() == 2

        assert analysisInstance.kineticStats.data["LHipMoment","Left"]["values"].__len__() == 2
        assert analysisInstance.kineticStats.data["RHipMoment","Right"]["values"].__len__() == 1

        acq = btkTools.smartReader(DATA_PATH+modelledFilenames[0])
        fig = plt.figure()
        ax = plt.gca()
        plot.temporalPlot(ax,acq,"LPelvisAngles",0,color="blue",
                title="test", xlabel="frame", ylabel="angle",ylim=None,legendLabel=None,
                customLimits=None)

        plt.show()

        fig = plt.figure()
        ax = plt.gca()
        plot.gaitDescriptivePlot(ax,analysisInstance.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color="blue",
                                title="", xlabel="", ylabel="",ylim=None,
                                customLimits=None)
        plt.show()

        fig = plt.figure()
        ax = plt.gca()
        plot.gaitConsistencyPlot(ax,analysisInstance.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color="blue",
                                title="", xlabel="", ylabel="",ylim=None,
                                customLimits=None)
        plt.show()
