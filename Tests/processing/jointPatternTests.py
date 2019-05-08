# -*- coding: utf-8 -*-
import ipdb
import logging
import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)


from pyCGM2.Processing import cycle,analysis,exporter,c3dManager
from pyCGM2.Tools import trialTools
from pyCGM2.Processing import jointPatterns


class DetectPatternTests():

    @classmethod
    def test(cls):
        criteria = "#1[normal]+#2[normal]"
        prim1,second1 = jointPatterns.JointPatternFilter.interpretCriteria(criteria)

        criteria = "(#3[decrease]|#4[decrease]|#5[delayed]|#6[decrease],2)"
        prim2,second2 = jointPatterns.JointPatternFilter.interpretCriteria(criteria)

        criteria = "#9[increase]+(#13[increase]|#10[early]|#14[excessive],2)"
        prim3,second3 = jointPatterns.JointPatternFilter.interpretCriteria(criteria)

        criteria = "#9[increase]+(#13[increase]|#10[early]|#14[excessive],2)+(#13[increase]|#10[early]|#14[excessive],2)"
        prim4,second4 = jointPatterns.JointPatternFilter.interpretCriteria(criteria)

    @classmethod
    def test_withData(cls):
        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\gaitDeviations\\gaitPig\\"
        modelledFilenames = ["gait Trial 01 - viconName.c3d","gait Trial 03 - viconName.c3d"  ]

        # ----INFOS-----
        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        normativeDataSet=dict()
        normativeDataSet["Author"] = "Schwartz2008"
        normativeDataSet["Modality"] = "Free"


        pointLabelSuffix=""

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

        pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix

        kinematicLabelsDict ={ 'Left': [str("LHipAngles"+pointLabelSuffixPlus),str("LKneeAngles"+pointLabelSuffixPlus),str("LAnkleAngles"+pointLabelSuffixPlus),str("LFootProgressAngles"+pointLabelSuffixPlus),str("LPelvisAngles"+pointLabelSuffixPlus)],
                               'Right': [str("RHipAngles"+pointLabelSuffixPlus),str("RKneeAngles"+pointLabelSuffixPlus),str("RAnkleAngles"+pointLabelSuffixPlus),str("RFootProgressAngles"+pointLabelSuffixPlus),str("RPelvisAngles"+pointLabelSuffixPlus)] }

        kineticLabelsDict ={ 'Left': [str("LHipMoment"+pointLabelSuffixPlus),str("LKneeMoment"+pointLabelSuffixPlus),str("LAnkleMoment"+pointLabelSuffixPlus), str("LHipPower"+pointLabelSuffixPlus),str("LKneePower"+pointLabelSuffixPlus),str("LAnklePower"+pointLabelSuffixPlus)],
                        'Right': [str("RHipMoment"+pointLabelSuffixPlus),str("RKneeMoment"+pointLabelSuffixPlus),str("RAnkleMoment"+pointLabelSuffixPlus), str("RHipPower"+pointLabelSuffixPlus),str("RKneePower"+pointLabelSuffixPlus),str("RAnklePower"+pointLabelSuffixPlus)]}


        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = kinematicLabelsDict,
                                                      kineticLabelsDict = kineticLabelsDict,
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=modelInfo,
                                                      experimentalInfos=experimentalInfo)

        analysisFilter = analysis.AnalysisFilter()
        analysisFilter.setBuilder(analysisBuilder)
        analysisFilter.build()


        #---- Joint patterns

        # xls Processing
        RULES_PATH = pyCGM2.PYCGM2_SETTINGS_FOLDER +"jointPatterns\\"
        rulesXls = RULES_PATH+"tests.xlsx"
        jpp = jointPatterns.XlsJointPatternProcedure(rulesXls)
        dpf = jointPatterns.JointPatternFilter(jpp, analysisFilter.analysis)
        dataFrameValues = dpf.getValues()
        dataFramePatterns = dpf.getPatterns()


class Nieuwenhuys2017_tests():

    @classmethod
    def kinematicsOnly_bothSide(cls):

        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\gaitDeviations\\gaitPig\\"
        modelledFilenames = ["gait Trial 01 - viconName.c3d","gait Trial 03 - viconName.c3d"  ]

        # ----INFOS-----
        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        normativeDataSet=dict()
        normativeDataSet["Author"] = "Schwartz2008"
        normativeDataSet["Modality"] = "Free"


        pointLabelSuffix=""

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

        pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix

        kinematicLabelsDict ={ 'Left': [str("LHipAngles"+pointLabelSuffixPlus),str("LKneeAngles"+pointLabelSuffixPlus),str("LAnkleAngles"+pointLabelSuffixPlus),str("LFootProgressAngles"+pointLabelSuffixPlus),str("LPelvisAngles"+pointLabelSuffixPlus)],
                               'Right': [str("RHipAngles"+pointLabelSuffixPlus),str("RKneeAngles"+pointLabelSuffixPlus),str("RAnkleAngles"+pointLabelSuffixPlus),str("RFootProgressAngles"+pointLabelSuffixPlus),str("RPelvisAngles"+pointLabelSuffixPlus)] }

        kineticLabelsDict ={ 'Left': [str("LHipMoment"+pointLabelSuffixPlus),str("LKneeMoment"+pointLabelSuffixPlus),str("LAnkleMoment"+pointLabelSuffixPlus), str("LHipPower"+pointLabelSuffixPlus),str("LKneePower"+pointLabelSuffixPlus),str("LAnklePower"+pointLabelSuffixPlus)],
                        'Right': [str("RHipMoment"+pointLabelSuffixPlus),str("RKneeMoment"+pointLabelSuffixPlus),str("RAnkleMoment"+pointLabelSuffixPlus), str("RHipPower"+pointLabelSuffixPlus),str("RKneePower"+pointLabelSuffixPlus),str("RAnklePower"+pointLabelSuffixPlus)]}


        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = kinematicLabelsDict,
                                                      kineticLabelsDict = kineticLabelsDict,
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=modelInfo,
                                                      experimentalInfos=experimentalInfo)

        analysisFilter = analysis.AnalysisFilter()
        analysisFilter.setBuilder(analysisBuilder)
        analysisFilter.build()


        #---- Joint patterns

        # xls Processing
        RULES_PATH = pyCGM2.PYCGM2_SETTINGS_FOLDER +"jointPatterns\\"
        rulesXls = RULES_PATH+"Nieuwenhuys2017.xlsx"
        jpp = jointPatterns.XlsJointPatternProcedure(rulesXls)
        dpf = jointPatterns.JointPatternFilter(jpp, analysisFilter.analysis)
        dataFrameValues = dpf.getValues()
        dataFramePatterns = dpf.getPatterns()

        xlsExport = exporter.XlsExportDataFrameFilter()
        xlsExport.setDataFrames([dataFrameValues])
        xlsExport.export("TestsPointData", path=DATA_PATH)

        xlsExport = exporter.XlsExportDataFrameFilter()
        xlsExport.setDataFrames([dataFramePatterns])
        xlsExport.export("TestsPatternsData", path=DATA_PATH)

if __name__ == "__main__":

    #DetectPatternTests.test()
    DetectPatternTests.test_withData()

    #plt.close("all")
    Nieuwenhuys2017_tests.kinematicsOnly_bothSide()
