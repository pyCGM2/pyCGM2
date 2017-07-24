
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:55:11 2017

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import pdb
import logging


import matplotlib.pyplot as plt
import pandas as pd

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# openMA
pyCGM2.CONFIG.addOpenma()
import ma.io
import ma.body

from pyCGM2.Report import plot,normativeDatabaseProcedure
from pyCGM2.Processing import cycle,analysis, discretePoints,exporter,c3dManager, automaticRules
from pyCGM2.Tools import trialTools

class AutomaticTest():

    @classmethod
    def test(cls):

        # ----DATA-----

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"operations\\analysis\\gait\\"
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



        # ----------- AUTOMATIC RULES -------------------------
        RULES_PATH = pyCGM2.CONFIG.MAIN_PYCGM2_PATH +"Data\\rules\\"
        rulesXls = RULES_PATH+"LowerLimb_AutomaticExtractionGaitDeviation.xlsx"

        arf = automaticRules.AutomaticExtractionFilter(rulesXls,analysisFilter.analysis)
        dataframe = arf.extract()

        xlsxWriter = pd.ExcelWriter(str(DATA_PATH +"dataframe-rules.xls"),engine='xlwt')
        dataframe.to_excel(xlsxWriter,"rule-results")
        xlsxWriter.save()

        print str(DATA_PATH +"dataframe-rules.xls")


if __name__ == "__main__":

    plt.close("all")

    AutomaticTest.test()


    #RULES_PATH = pyCGM2.CONFIG.MAIN_PYCGM2_PATH +"Data\\rules\\"
    #rulesXls = RULES_PATH+"LowerLimb_AutomaticExtractionGaitDeviation.xlsx"
    #rules = pd.read_excel(rulesXls)
    # ipdb.set_trace()
