# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:54:18 2017

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import ipdb
import logging


import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)


from pyCGM2.Processing import cycle,analysis,scores,exporter,c3dManager
from pyCGM2.Report import normativeDatasets
from pyCGM2.Tools import trialTools




class GpsTest():

    @classmethod
    def GpsCGM1Test(cls):
         # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gps\\"

        reconstructedFilenameLabelledNoExt ="gait Trial 03 - viconName"
        reconstructedFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"

        logging.info("data Path: "+ DATA_PATH)
        logging.info( "reconstructed file: "+ reconstructedFilenameLabelled)


        modelledFilenames = [reconstructedFilenameLabelled]

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

        ## --- GPS ----
        ndp = normativeDatasets.Schwartz2008("Free")

        gps =scores.CGM1_GPS()
        scf = scores.ScoreFilter(gps,analysisInstance, ndp)
        scf.compute()

        xlsExport = exporter.XlsAnalysisExportFilter()
        xlsExport.setAnalysisInstance(analysisInstance)
        xlsExport.export("gpsTest2", path=DATA_PATH, mode="Advanced")






if __name__ == "__main__":

    plt.close("all")

    GpsTest.GpsCGM1Test()
