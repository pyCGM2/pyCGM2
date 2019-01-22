# -*- coding: utf-8 -*-
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

class oneTrial_PlotTest():

    @classmethod
    def temporalPlot_OneModelOutputPlot(cls):
        """
        Plot only one Model output
        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]

        trial =trialTools.smartTrialReader(DATA_PATH,modelledFilenames[0])

        fig = plt.figure()
        ax = plt.gca()
        plot.temporalPlot(ax,trial,"LPelvisAngles",0,color="blue",
                title="test", xlabel="frame", ylabel="angle",ylim=None,legendLabel=None,
                customLimits=None)

        plt.show()

class oneAnalysis_PlotTest():
    @classmethod
    def descriptivePlot_OneModelOutputPlot(cls):
        """
        Plot only one Model output
        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]


        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None


        analysisInstance = analysis.makeAnalysis("Gait", "CGM1.0", DATA_PATH,modelledFilenames,None, None, None)

        fig = plt.figure()
        ax = plt.gca()
        plot.gaitDescriptivePlot(ax,analysisInstance.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color="blue",
                                title="", xlabel="", ylabel="",ylim=None,
                                customLimits=None)
        plt.show()

    @classmethod
    def consistencyPlot_OneModelOutputPlot(cls):
        """
        Plot only one Model output
        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]


        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisInstance = analysis.makeAnalysis("Gait", "CGM1.0", DATA_PATH,modelledFilenames,None, None, None)


        fig = plt.figure()
        ax = plt.gca()
        plot.gaitConsistencyPlot(ax,analysisInstance.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color="blue",
                                title="", xlabel="", ylabel="",ylim=None,
                                customLimits=None)
        plt.show()

    @classmethod
    def meanPlot_OneModelOutputPlot(cls):
        """
        Plot only one Model output
        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]


        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisInstance = analysis.makeAnalysis("Gait", "CGM1.0", DATA_PATH,modelledFilenames,None, None, None)

        fig = plt.figure()
        ax = plt.gca()
        plot.gaitMeanPlot(ax,analysisInstance.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color="blue",
                                title="", xlabel="", ylabel="",ylim=None,
                                customLimits=None)
        plt.show()


class multipleAnalysis_PlotTest():
    @classmethod
    def consistencyPlot_OneModelOutputPlot(cls):
        """

        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]


        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysis1 = analysis.makeAnalysis("Gait", "CGM1.0", DATA_PATH,modelledFilenames,None, None, None)

        analysis2 = analysis.makeAnalysis("Gait", "CGM1.0", DATA_PATH,modelledFilenames,None, None, None)

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
        plt.show()

    @classmethod
    def meanPlot_OneModelOutputPlot(cls):
        """
        Plot only one Model output
        """

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]


        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysis1 = analysis.makeAnalysis("Gait", "CGM1.0", DATA_PATH,modelledFilenames,None, None, None)

        analysis2 = analysis.makeAnalysis("Gait", "CGM1.0", DATA_PATH,modelledFilenames,None, None, None)

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
        plt.show()

if __name__ == "__main__":

    plt.close("all")

    oneTrial_PlotTest.temporalPlot_OneModelOutputPlot()
    oneAnalysis_PlotTest.descriptivePlot_OneModelOutputPlot()
    oneAnalysis_PlotTest.consistencyPlot_OneModelOutputPlot()
    oneAnalysis_PlotTest.meanPlot_OneModelOutputPlot()

    multipleAnalysis_PlotTest.consistencyPlot_OneModelOutputPlot()
    multipleAnalysis_PlotTest.meanPlot_OneModelOutputPlot()
