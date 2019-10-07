# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_gaitPlotPanelTests.py::Test_oneAnalysis_GaitPlotTest::test_gaitDescriptiveKinematicPlotPanel
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


class Test_oneAnalysis_StandardPlotTest():


    def test_descriptiveKinematicPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot_latin1_çà\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]

        # #---- c3d manager
        # #--------------------------------------------------------------------------
        # c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        # cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        # cmf.enableEmg(False)
        # trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------

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
        pf.plot()

        plt.show()






class Test_oneAnalysis_GaitPlotTest():


    def test_gaitDescriptiveKinematicPlotPanel(self):

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
        kv = plotViewers.NormalizedKinematicsPlotViewer(analysisInstance)
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        plt.show()


    def test_gaitDescriptiveKinematicPlotPanel_recorded(self):

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
        kv = plotViewers.NormalizedKinematicsPlotViewer(analysisInstance)
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.setExport(DATA_PATH,"check","png")
        pf.plot()

        plt.show()


    def test_gaitConsistencyKinematicPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]


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

        plt.show()


    def test_gaitDescriptiveKineticPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]

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

        plt.show()

class Test_multipleAnalysis_GaitPlotTest():


    def test_gaitDescriptiveKinematicPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]


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

        plt.show()



    def test_gaitConsistencyKinematicPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]

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

        plt.show()


    def test_gaitMeanOnlyKinematicPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]

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

        plt.show()


    def test_gaitConsistencyKineticPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]



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

        plt.show()


    def test_gaitMeanOnlyKineticPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]



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

        plt.show()
