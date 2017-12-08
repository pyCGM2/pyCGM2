# -*- coding: utf-8 -*-
import logging

import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2

from pyCGM2.Processing.gaitAnalysis import smartFunctions
from pyCGM2 import enums
from pyCGM2.Processing import c3dManager
from pyCGM2.Model.CGM2 import  cgm,cgm2

from pyCGM2.Report import plot,plotFilters,plotViewers,normativeDatasets
from pyCGM2.Tools import trialTools

class oneAnalysis_PlotTest():

    @classmethod
    def descriptiveKinematicPlotPanel(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
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

        analysis = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)

        # viewer
        kv = plotViewers.GaitKinematicsPlotViewer(analysis)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        plt.show()

    @classmethod
    def descriptiveKinematicPlotPanel_recorded(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
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

        analysis = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)

        # viewer
        kv = plotViewers.GaitKinematicsPlotViewer(analysis)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.setPath("C:\\Users\\HLS501\\Desktop\\")
        pf.setPdfName("test")
        pf.plot()

        plt.show()

    @classmethod
    def consistencyKinematicPlotPanel(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
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

        analysis = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)

        # viewer
        kv = plotViewers.GaitKinematicsPlotViewer(analysis,plotType = enums.PlotType.CONSISTENCY)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        plt.show()

    @classmethod
    def descriptiveKineticPlotPanel(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
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

        analysis = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)

        # viewer
        if trialManager.kineticFlag:
            kv = plotViewers.GaitKineticsPlotViewer(analysis)
            kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

            # filter
            pf = plotFilters.PlottingFilter()
            pf.setViewer(kv)
            pf.plot()

        plt.show()

class multipleAnalysis_PlotTest():

    @classmethod
    def consistencyKinematicPlotPanel(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
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


        analysis1 = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)

        analysis2 = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)

        kv = plotViewers.multipleAnalyses_GaitKinematicsPlotViewer([analysis1,analysis2],"Left",["ana1","ana2"])
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        plt.show()

    @classmethod
    def meanOnlyKinematicPlotPanel(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
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

        analysis1 = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)

        analysis2 = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)

        kv = plotViewers.multipleAnalyses_GaitKinematicsPlotViewer([analysis1,analysis2],"Left",
                                                                    ["ana1","ana2"],
                                    plotType=enums.PlotType.MEAN_ONLY)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        plt.show()


    @classmethod
    def consistencyKineticPlotPanel(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
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

        analysis1 = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)

        analysis2 = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)


        kv = plotViewers.multipleAnalyses_GaitKineticsPlotViewer([analysis1,analysis2],"Right",
                                                                    ["ana1","ana2"],
                                    plotType=enums.PlotType.CONSISTENCY)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        plt.show()

    @classmethod
    def meanOnlyKineticPlotPanel(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
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

        analysis1 = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)

        analysis2 = smartFunctions.make_analysis(trialManager,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                                                cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                                    modelInfo,subjectInfo,experimentalInfo)


        kv = plotViewers.multipleAnalyses_GaitKineticsPlotViewer([analysis1,analysis2],"Right",
                                                                    ["ana1","ana2"],
                                    plotType=enums.PlotType.MEAN_ONLY)
        kv.setNormativeDataset(normativeDatasets.Schwartz2008("Free"))

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        plt.show()

if __name__ == "__main__":

    plt.close("all")

    oneAnalysis_PlotTest.descriptiveKinematicPlotPanel()
    oneAnalysis_PlotTest.descriptiveKinematicPlotPanel_recorded()
    oneAnalysis_PlotTest.consistencyKinematicPlotPanel()
    oneAnalysis_PlotTest.descriptiveKineticPlotPanel()

    multipleAnalysis_PlotTest.consistencyKinematicPlotPanel()
    multipleAnalysis_PlotTest.meanOnlyKinematicPlotPanel()
    multipleAnalysis_PlotTest.consistencyKineticPlotPanel()
    multipleAnalysis_PlotTest.meanOnlyKineticPlotPanel()
