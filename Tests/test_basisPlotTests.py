# -*- coding: utf-8 -*-
# pytest --disable-pytest-warnings  test_basisPlotTests.py::Test_oneTrial_PlotTest::test_temporalPlot_OneModelOutputPlot
import logging
import numpy as np
import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
from pyCGM2.Lib import analysis

from pyCGM2.Model.CGM2 import  cgm,cgm2
from pyCGM2.Processing import c3dManager

from pyCGM2.Report import plot
from pyCGM2.Tools import trialTools


class Test_oneTrial_PlotTest():


    def test_temporalPlot_OneModelOutputPlot(self):
        """
        Plot only one Model output
        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d"]

        trial =trialTools.smartTrialReader(DATA_PATH,modelledFilenames[0])

        fig = plt.figure()
        ax = plt.gca()
        plot.temporalPlot(ax,trial,"LPelvisAngles",0,color="blue",
                title="test", xlabel="frame", ylabel="angle",ylim=None,legendLabel=None,
                customLimits=None)

        #plt.show()

class Test_oneAnalysis_PlotTest():

    def test_descriptivePlot_OneModelOutputPlot(self):
        """
        Plot only one Model output
        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None


        analysisInstance = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        fig = plt.figure()
        ax = plt.gca()
        plot.gaitDescriptivePlot(ax,analysisInstance.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color="blue",
                                title="", xlabel="", ylabel="",ylim=None,
                                customLimits=None)
        #plt.show()


    def test_consistencyPlot_OneModelOutputPlot(self):
        """
        Plot only one Model output
        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]


        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisInstance = analysis.makeAnalysis(DATA_PATH,modelledFilenames)


        fig = plt.figure()
        ax = plt.gca()
        plot.gaitConsistencyPlot(ax,analysisInstance.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color="blue",
                                title="", xlabel="", ylabel="",ylim=None,
                                customLimits=None)
        #plt.show()


    def test_meanPlot_OneModelOutputPlot(self):
        """
        Plot only one Model output
        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisInstance = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        fig = plt.figure()
        ax = plt.gca()
        plot.gaitMeanPlot(ax,analysisInstance.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color="blue",
                                title="", xlabel="", ylabel="",ylim=None,
                                customLimits=None)
        #plt.show()


class Test_multipleAnalysis_PlotTest():

    def test_consistencyPlot_OneModelOutputPlot(self):
        """

        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames1 = ["gait Trial 01.c3d"]
        modelledFilenames2 = ["gait Trial 02.c3d"]

        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysis1 = analysis.makeAnalysis(DATA_PATH,modelledFilenames1)
        analysis2 = analysis.makeAnalysis(DATA_PATH,modelledFilenames2)

        analyses = [analysis1, analysis2]
        fig = plt.figure()
        ax = plt.gca()
        colormap = plt.cm.Reds
        colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(analyses))]

        plot.gaitConsistencyPlot(ax,analysis1.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color=colormap_i[0],
                                legendLabel="analysis1",
                                customLimits=None)

        plot.gaitConsistencyPlot(ax,analysis2.kinematicStats,
                                 "LKneeAngles","Left",0,
                                 color=colormap_i[1],
                                 legendLabel="analysis2",
                                 customLimits=None)
        ax.legend()
        #plt.show()


    def test_meanPlot_OneModelOutputPlot(self):
        """
        Plot only one Model output
        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames1 = ["gait Trial 01.c3d"]
        modelledFilenames2 = ["gait Trial 02.c3d"]


        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysis1 = analysis.makeAnalysis(DATA_PATH,modelledFilenames1)
        analysis2 = analysis.makeAnalysis(DATA_PATH,modelledFilenames2)

        analyses = [analysis1, analysis2]
        fig = plt.figure()
        ax = plt.gca()
        colormap = plt.cm.Reds
        colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(analyses))]

        plot.gaitMeanPlot(ax,analysis1.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color=colormap_i[0],
                                legendLabel="analysis1",
                                customLimits=None)

        plot.gaitMeanPlot(ax,analysis2.kinematicStats,
                                 "LKneeAngles","Left",0,
                                 color=colormap_i[1],
                                 legendLabel="analysis2",
                                 customLimits=None)
        ax.legend()
        #plt.show()
