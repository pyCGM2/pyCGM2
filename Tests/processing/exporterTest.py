# -*- coding: utf-8 -*-
import ipdb
import matplotlib.pyplot as plt
import logging

import pyCGM2
from pyCGM2 import Lib


from pyCGM2.Model.CGM2 import cgm

from pyCGM2.Processing import exporter,c3dManager,cycle,analysis




class ExportTest():

    @classmethod
    def analysisAdvanced(cls):

        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d" ]

        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------

        # ----INFOS-----
        modelInfo={"Type":"cgm2", "hjc":"hara"}
        subjectInfo={"Id":"1", "Name":"Lecter"}
        experimentalInfo={"Condition":"Barefoot", "context":"block"}

        analysisInstance = Lib.analysis.makeAnalysis("Gait", "CGM1.0", DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)


        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(analysisInstance)
        exportFilter.export("testAdvanced", path=DATA_PATH,excelFormat = "xls",mode="Advanced")

    @classmethod
    def analysisAdvancedAndBasic_nonExistingLabel(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d","gait Trial 01 - viconName.c3d" ]

        #---- c3d manager
        #--------------------------------------------------------------------------

        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()



        #---- GAIT CYCLES FILTER
        #--------------------------------------------------------------------------
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalTrials=trialManager.spatioTemporal["Trials"],
                                                   kinematicTrials = trialManager.kinematic["Trials"],
                                                   kineticTrials = trialManager.kinetic["Trials"],
                                                   emgTrials=trialManager.emg["Trials"])

        cyclefilter = cycle.CyclesFilter()
        cyclefilter.setBuilder(cycleBuilder)
        cycles = cyclefilter.build()


        #---- GAIT ANALYSIS FILTER
        #--------------------------------------------------------------------------

        # ----INFOS-----
        modelInfo={"type":"S01"}
        subjectInfo=None
        experimentalInfo=None

        kinematicLabelsDict ={ 'Left': ["LHipAngles","LKneeAngles","LAnkleAngles","LForeFootAngles"],
                        'Right': ["RHipAngles","RKneeAngles","RAnkleAngles"] }

        kineticLabelsDict =None#{ 'Left': ["LHipMoment","LKneeMoment"],
                             #'Right': ["RHipMoment","RKneeMoment"]}


        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = kinematicLabelsDict,
                                                      kineticLabelsDict = kineticLabelsDict,
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=modelInfo,
                                                      experimentalInfos=experimentalInfo)

        analysisFilter = analysis.AnalysisFilter()
        analysisFilter.setBuilder(analysisBuilder)
        analysisFilter.setInfo(model = modelInfo)
        analysisFilter.build()

        analysisInstance = analysisFilter.analysis

        # ----INFOS-----
        modelInfo={"Type":"cgm2", "hjc":"hara"}
        subjectInfo={"Id":"1", "Name":"Lecter"}
        experimentalInfo={"Condition":"Barefoot", "context":"block"}



        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(analysisInstance)
        exportFilter.export("testAdvancedNan", path=DATA_PATH,excelFormat = "xls",mode="Advanced")

        exportFilter2 = exporter.XlsAnalysisExportFilter()
        exportFilter2.setAnalysisInstance(analysisInstance)
        exportFilter2.export("testBasic2", path=DATA_PATH,excelFormat = "xls",mode="Basic")

    @classmethod
    def analysisBasic(cls):

        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d" ]

        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- Analysis
        #--------------------------------------------------------------------------

        # ----INFOS-----
        modelInfo={"Type":"cgm2", "hjc":"hara"}
        subjectInfo={"Id":"1", "Name":"Lecter"}
        experimentalInfo={"Condition":"Barefoot", "context":"block"}


        analysisInstance = Lib.analysis.makeAnalysis("Gait", "CGM1.0", DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)


        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(analysisInstance)
        exportFilter.export("test", path=DATA_PATH,excelFormat = "xls",mode="Basic")


if __name__ == "__main__":

    plt.close("all")

    ExportTest.analysisAdvanced()
    ExportTest.analysisAdvancedAndBasic_nonExistingLabel()
    ExportTest.analysisBasic()
