# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_gaitPlotPanelTests.py::Test_oneAnalysis_StandardPlotTest::test_descriptiveKinematicPlotPanel
import logging

import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
from pyCGM2.Lib import analysis

from pyCGM2 import enums
from pyCGM2.Processing import c3dManager
from pyCGM2.Model.CGM2 import  cgm,cgm2

from pyCGM2.Report import plot,plotFilters,plotViewers,normativeDatasets,ComparisonPlotViewers
from pyCGM2.Tools import trialTools
from pyCGM2.Report import plot
from pyCGM2.Utils import files


class Test_oneAnalysis_StandardPlotTest():


    def test_descriptiveKinematicPlotPanel_savePngAndPdf(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        files.createDir(DATA_PATH_OUT)

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisInstance = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        # viewer
        kv = plotViewers.NormalizedKinematicsPlotViewer(analysisInstance)
        kv.setConcretePlotFunction(plot.descriptivePlot)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.setExport(DATA_PATH_OUT,"test_descriptiveKinematicPlotPanel","png")
        pf.plot()

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.setExport(DATA_PATH_OUT,"test_descriptiveKinematicPlotPanel","pdf")
        pf.plot()






class Test_oneAnalysis_GaitPlotTest():


    def test_gaitDescriptiveKinematicPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]


        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisInstance = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        # viewer
        kv = plotViewers.NormalizedKinematicsPlotViewer(analysisInstance)
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        #plt.show()

    def test_gaitConsistencyKinematicPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]


        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisInstance = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        # viewer
        kv = plotViewers.NormalizedKinematicsPlotViewer(analysisInstance)
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        #plt.show()


    def test_gaitDescriptiveKineticPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisInstance = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        # viewer
        kv = plotViewers.NormalizedKineticsPlotViewer(analysisInstance)
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        #plt.show()

class Test_multipleAnalysis_GaitPlotTest():


    def test_gaitDescriptiveKinematicPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]


        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None


        analysis1 = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        analysis2 = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        kv = ComparisonPlotViewers.KinematicsPlotComparisonViewer([analysis1,analysis2],"Left",["ana1","ana2"])
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        #plt.show()



    def test_gaitConsistencyKinematicPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None


        analysis1 = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        analysis2 = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        kv = ComparisonPlotViewers.KinematicsPlotComparisonViewer([analysis1,analysis2],"Left",["ana1","ana2"])
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        #plt.show()


    def test_gaitMeanOnlyKinematicPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysis1 = analysis.makeAnalysis(DATA_PATH,modelledFilenames)
        analysis2 = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        kv = ComparisonPlotViewers.KinematicsPlotComparisonViewer([analysis1,analysis2],"Left",
                                                                    ["ana1","ana2"])
        kv.setConcretePlotFunction(plot.gaitMeanPlot)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        #plt.show()


    def test_gaitConsistencyKineticPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]



        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysis1 = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        analysis2 = analysis.makeAnalysis(DATA_PATH,modelledFilenames)


        kv = ComparisonPlotViewers.KineticsPlotComparisonViewer([analysis1,analysis2],"Right",
                                                                    ["ana1","ana2"])
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        #plt.show()


    def test_gaitMeanOnlyKineticPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"GaitData\CGM1-NormalGaitData-Events\Hånnibøl Lecter\\"
        modelledFilenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]



        #---- Analysis
        #--------------------------------------------------------------------------

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysis1 = analysis.makeAnalysis(DATA_PATH,modelledFilenames)

        analysis2 = analysis.makeAnalysis(DATA_PATH,modelledFilenames)


        kv = ComparisonPlotViewers.KineticsPlotComparisonViewer([analysis1,analysis2],"Right",
                                                                    ["ana1","ana2"])
        kv.setConcretePlotFunction(plot.gaitMeanPlot)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        #plt.show()
