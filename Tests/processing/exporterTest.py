# -*- coding: utf-8 -*-
import ipdb
import matplotlib.pyplot as plt
import logging

import pyCGM2
import pyCGM2.Lib.analysis


from pyCGM2.Model.CGM2 import cgm

from pyCGM2.Processing import exporter,c3dManager,cycle,analysis
from pyCGM2.Tools import btkTools



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

        analysisInstance = pyCGM2.Lib.analysis.makeAnalysis("Gait",  DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)


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


        analysisInstance = pyCGM2.Lib.analysis.makeAnalysis("Gait", DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)


        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(analysisInstance)
        exportFilter.export("test", path=DATA_PATH,excelFormat = "xls",mode="Basic")


    @classmethod
    def analysisJson(cls):

        # ----DATA-----

        #DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\analysis\gait\\"
        modelledFilenames = ["gait Trial 01 - viconName.c3d", "gait Trial 03 - viconName.c3d" ]

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

        analysisInstance = pyCGM2.Lib.analysis.makeAnalysis("Gait", DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)


        exportFilter = exporter.AnalysisExportFilter()
        exportFilter.setAnalysisInstance(analysisInstance)
        exportFilter.export("testAdvanced", path=DATA_PATH)

    @classmethod
    def analysisC3d(cls):

        # ----DATA-----

        #DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\analysis\gait\\"
        modelledFilenames = ["gait Trial 01 - viconName.c3d", "gait Trial 03 - viconName.c3d" ]

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

        analysisInstance = pyCGM2.Lib.analysis.makeAnalysis("Gait", DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)


        exportFilter = exporter.AnalysisC3dExportFilter()
        exportFilter.setAnalysisInstance(analysisInstance)
        exportFilter.export("testAdvanced", path=DATA_PATH)

    @classmethod
    def analysisAdvancedEMG(cls):

        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gaitEMG\\"
        inputFile = ["pre.c3d","post.c3d"]

        EMG_LABELS = ["EMG1","EMG2"]
#
        for file in inputFile:
            acq = btkTools.smartReader(DATA_PATH+file)
            pyCGM2.Lib.analysis.processEMG(acq, EMG_LABELS, highPassFrequencies=[20,200],envelopFrequency=6.0)
            btkTools.smartWriter(acq,DATA_PATH+file[:-4]+"-emgProcessed.c3d")

        inputFileProcessed =   [file[:-4]+"-emgProcessed.c3d" for file in inputFile]

        emgAnalysis =  pyCGM2.Lib.analysis.makeEmgAnalysis(DATA_PATH, inputFileProcessed, EMG_LABELS,None, None)

        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysis)
        exportFilter.export("testAdvancedEMG", path=DATA_PATH,excelFormat = "xls",mode="Advanced")

        # exportFilter = exporter.AnalysisExportFilter()
        # exportFilter.setAnalysisInstance(emgAnalysis)
        # exportFilter.export("testAdvancedEMG.json", path=DATA_PATH)

        exportFilter = exporter.AnalysisC3dExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysis)
        exportFilter.export("testAdvanced", path=DATA_PATH)

if __name__ == "__main__":

    plt.close("all")

    #ExportTest.analysisAdvanced()
    # ExportTest.analysisAdvancedAndBasic_nonExistingLabel()
    # ExportTest.analysisBasic()
    #ExportTest.analysisJson()
    #ExportTest.analysisC3d()
    ExportTest.analysisAdvancedEMG()
