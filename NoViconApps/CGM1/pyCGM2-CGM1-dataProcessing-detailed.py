# -*- coding: utf-8 -*-

import os
import logging
import matplotlib.pyplot as plt
import json
import pdb
import cPickle
import json
from collections import OrderedDict
from shutil import copyfile
import argparse


# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# openMA
#import ma.io
#import ma.body

#btk
import btk


# pyCGM2 libraries
from pyCGM2 import  smartFunctions

from pyCGM2.Processing import cycle,analysis,scores,exporter,c3dManager, discretePoints
from pyCGM2.Report import plot,normativeDatabaseProcedure

#

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gait Processing')
    parser.add_argument('--pointSuffix', type=str, help='force suffix')
    args = parser.parse_args()

    # --------------------SESSIONS INFOS ------------------------------

    infoSettings = json.loads(open('pyCGM2.info').read(),object_pairs_hook=OrderedDict)

    if args.pointSuffix is not None:
        pointSuffix = args.pointSuffix
    else:
        pointSuffix = infoSettings["Processing"]["Point suffix"]

    normativeData = infoSettings["Processing"]["Normative data"]

    # -----infos--------
    model = None if  infoSettings["Modelling"]["Model"]=={} else infoSettings["Modelling"]["Model"]
    subject = None if infoSettings["Processing"]["Subject"]=={} else infoSettings["Processing"]["Subject"]
    experimental = None if infoSettings["Processing"]["Experimental conditions"]=={} else infoSettings["Processing"]["Experimental conditions"]

    # --------------------------PROCESSING --------------------------------

    DATA_PATH = infoSettings["Modelling"]["Trials"]["DataPath"]
    motionTrialFilenames = infoSettings["Modelling"]["Trials"]["Motion"]

    #---- NORMATIVE DATASET
    #--------------------------------------------------------------------------

    if normativeData["Author"] == "Schwartz2008":
        chosenModality = normativeData["Modality"]
        ndp = normativeDatabaseProcedure.Schwartz2008_normativeDataBases(chosenModality)    # modalites : "Very Slow" ,"Slow", "Free", "Fast", "Very Fast"
    elif normativeData["Author"] == "Pinzone2014":
        chosenModality = normativeData["Modality"]
        ndp = normativeDatabaseProcedure.Pinzone2014_normativeDataBases(chosenModality) # modalites : "Center One" ,"Center Two"

    #---- Modelled File manager
    #--------------------------------------------------------------------------
    # preliminary check if modelledFilenames is string
    if isinstance(motionTrialFilenames,str) or isinstance(motionTrialFilenames,unicode):
        logging.info( "gait Processing on ONE file")
        modelledFilenames = [motionTrialFilenames]
    else:
        modelledFilenames = motionTrialFilenames


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

    pointLabelSuffixPlus  = pointSuffix   if pointSuffix =="" else "_"+pointSuffix

    kinematicLabelsDict ={ 'Left': [str("LHipAngles"+pointLabelSuffixPlus),str("LKneeAngles"+pointLabelSuffixPlus),str("LAnkleAngles"+pointLabelSuffixPlus),str("LFootProgressAngles"+pointLabelSuffixPlus),str("LPelvisAngles"+pointLabelSuffixPlus)],
                           'Right': [str("RHipAngles"+pointLabelSuffixPlus),str("RKneeAngles"+pointLabelSuffixPlus),str("RAnkleAngles"+pointLabelSuffixPlus),str("RFootProgressAngles"+pointLabelSuffixPlus),str("RPelvisAngles"+pointLabelSuffixPlus)] }

    kineticLabelsDict ={ 'Left': [str("LHipMoment"+pointLabelSuffixPlus),str("LKneeMoment"+pointLabelSuffixPlus),str("LAnkleMoment"+pointLabelSuffixPlus), str("LHipPower"+pointLabelSuffixPlus),str("LKneePower"+pointLabelSuffixPlus),str("LAnklePower"+pointLabelSuffixPlus)],
                    'Right': [str("RHipMoment"+pointLabelSuffixPlus),str("RKneeMoment"+pointLabelSuffixPlus),str("RAnkleMoment"+pointLabelSuffixPlus), str("RHipPower"+pointLabelSuffixPlus),str("RKneePower"+pointLabelSuffixPlus),str("RAnklePower"+pointLabelSuffixPlus)]}


    analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                  kinematicLabelsDict = kinematicLabelsDict,
                                                  kineticLabelsDict = kineticLabelsDict,
                                                  subjectInfos=subject,
                                                  modelInfos=model,
                                                  experimentalInfos=experimental)

    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()


    # --- GPS ----
    gps =scores.CGM1_GPS()
    scf = scores.ScoreFilter(gps,analysisFilter.analysis, ndp)
    scf.compute()

    # --- DiscretePoint ----

    # Benedetti Processing
    dpProcedure = discretePoints.BenedettiProcedure()
    dpf = discretePoints.DiscretePointsFilter(dpProcedure, analysisFilter.analysis)
    benedettiDataFrame = dpf.getOutput()

    # MaxMin Processing
    dpProcedure = discretePoints.MaxMinProcedure()
    dpf = discretePoints.DiscretePointsFilter(dpProcedure, analysisFilter.analysis)
    maxMinDataFrame = dpf.getOutput()


    # --- Excel exporter ----

    spreadSheetName= "Session"
    xlsExport = exporter.XlsExportFilter()
    xlsExport.setDataFrames([benedettiDataFrame,maxMinDataFrame])
    xlsExport.exportDataFrames("discretePoints", path=DATA_PATH)

    xlsExport.setAnalysisInstance(analysisFilter.analysis)
    xlsExport.setConcreteAnalysisBuilder(analysisBuilder)
    xlsExport.exportAdvancedDataFrame(spreadSheetName, path=DATA_PATH)

    #---- GAIT PLOTTING FILTER
    #--------------------------------------------------------------------------

    plotBuilder = plot.GaitAnalysisPlotBuilder(analysisFilter.analysis , kineticFlag=trialManager.kineticFlag, pointLabelSuffix= pointSuffix)
    plotBuilder.setNormativeDataProcedure(ndp)
    plotBuilder.setConsistencyOnly(True)

    # Filter
    pdfName = "session"
    pf = plot.PlottingFilter()
    pf.setBuilder(plotBuilder)
    pf.setPath(DATA_PATH)
    pf.setPdfName(pdfName)
    pf.plot()


    os.startfile(DATA_PATH+"consistencyKinematics_"+ pdfName +".pdf")
    if trialManager.kineticFlag: os.startfile(DATA_PATH+"consistencyKinetics_"+ pdfName+".pdf")
