# -*- coding: utf-8 -*-
import ipdb
import matplotlib.pyplot as plt
import logging

import pyCGM2
from pyCGM2 import Lib

from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Processing import exporter,c3dManager,cycle,analysis
from pyCGM2.Utils import files


class AnalysisTest():


    @classmethod
    def detailedProcess(cls):
        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d" ]

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

        pointLabelSuffix =""
        pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix

        kinematicLabelsDict ={ 'Left': ["LHipAngles"+pointLabelSuffixPlus,"LKneeAngles"+pointLabelSuffixPlus,"LAnkleAngles"+pointLabelSuffixPlus,"LFootProgressAngles"+pointLabelSuffixPlus,"LPelvisAngles"+pointLabelSuffixPlus],
                        'Right': ["RHipAngles"+pointLabelSuffixPlus,"RKneeAngles"+pointLabelSuffixPlus,"RAnkleAngles"+pointLabelSuffixPlus,"RFootProgressAngles"+pointLabelSuffixPlus,"RPelvisAngles"+pointLabelSuffixPlus] }

        kineticLabelsDict ={ 'Left': ["LHipMoment"+pointLabelSuffixPlus,"LKneeMoment"+pointLabelSuffixPlus,"LAnkleMoment"+pointLabelSuffixPlus, "LHipPower"+pointLabelSuffixPlus,"LKneePower"+pointLabelSuffixPlus,"LAnklePower"+pointLabelSuffixPlus],
                        'Right': ["RHipMoment"+pointLabelSuffixPlus,"RKneeMoment"+pointLabelSuffixPlus,"RAnkleMoment"+pointLabelSuffixPlus, "RHipPower"+pointLabelSuffixPlus,"RKneePower"+pointLabelSuffixPlus,"RAnklePower"+pointLabelSuffixPlus]}


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


    @classmethod
    def detailedProcess_labelNotInTrial(cls):
        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gait\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d" ]

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




    @classmethod
    def makeAnalysis_oneFile_noInfo(cls):

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

        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisInstance = Lib.analysis.makeAnalysis("Gait",  DATA_PATH,modelledFilenames,None, None, None)


    @classmethod
    def makeAnalysis_oneFile_withInfo(cls):

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

        analysisInstance = Lib.analysis.makeAnalysis("Gait",  DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)


    @classmethod
    def Save_and_openAnalysis(cls):
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

        analysisInstance = Lib.analysis.makeAnalysis("Gait",  DATA_PATH,modelledFilenames,subjectInfo, experimentalInfo, modelInfo)

        files.saveAnalysis(analysisInstance,DATA_PATH,"Save_and_openAnalysis")
        analysis2 = files.loadAnalysis(DATA_PATH,"Save_and_openAnalysis")

if __name__ == "__main__":

    plt.close("all")

    AnalysisTest.detailedProcess_labelNotInTrial()
    AnalysisTest.makeAnalysis_oneFile_noInfo()
    AnalysisTest.makeAnalysis_oneFile_withInfo()
    AnalysisTest.Save_and_openAnalysis()
