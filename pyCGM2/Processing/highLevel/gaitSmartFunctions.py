# -*- coding: utf-8 -*-
#import ipdb
import logging

from  pyCGM2 import enums
from pyCGM2.Processing import cycle,analysis,scores,exporter,c3dManager
from pyCGM2.Report import plot,plotFilters,plotViewers,normativeDatasets
from pyCGM2.Tools import trialTools
from pyCGM2.Report import plot

# openma
import ma.io


def make_analysis(trialManager, kinematicLabelsDict,kineticLabelsDict,
                  modelInfo, subjectInfo, experimentalInfo,
                  pointLabelSuffix=""):
    """
    build a Analysis instance

    :Parameters:
       - `cgmSkeletonEnum` (pyCGM2.enums) - type of skeleton used
       - `trialManager` (pyCGM2.Processing.c3dManager.C3dManager) - organization of inputed c3d
       - `modelInfo` (dict) - info about the model
       - `subjectInfo` (dict) -  info about the subject
       - `experimentalInfo` (dict) - info about experimental conditions
       - `pointLabelSuffix` (str) - suffix added to standard cgm nomenclature

    :Return:
       - `NoName` (pyCGM2.Processing.analysis.Analysis) - Analysis instance

    """


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

    analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                  kinematicLabelsDict = kinematicLabelsDict,
                                                  kineticLabelsDict = kineticLabelsDict,
                                                  pointlabelSuffix = pointLabelSuffix)

    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setInfo(subject=subjectInfo, model=modelInfo, experimental=experimentalInfo)
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()

    return analysisFilter.analysis

def cgm_staticPlot(modelVersion,modelledStaticFilename, DATA_PATH,
    pdfFilename="staticProcessing",pointLabelSuffix=""):

    # check model is the CGM1
    logging.info("loaded model : %s" %(modelVersion ))

    trial =trialTools.smartTrialReader(DATA_PATH,modelledStaticFilename)

    #viewer
    if modelVersion in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e","CGM2.3","CGM2.3e"]:
        kv = plotViewers.TemporalGaitKinematicsPlotViewer(trial,pointLabelSuffix=pointLabelSuffix)
    else:
        raise Exception("[pyCGM2] Model version not known")

    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    pf.setPath(DATA_PATH)
    pf.setPdfName(pdfFilename)
    pf.plot()

def cgm_gaitPlots(modelVersion,analysis,kineticFlag,
    DATA_PATH,pdfFilename,
    pointLabelSuffix="",
    normativeDataset=None ):

    # filter 1 - descriptive kinematic panel
    #-------------------------------------------
    # viewer
    if modelVersion in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e","CGM2.3","CGM2.3e"]:
        kv = plotViewers.LowerLimbKinematicsPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix)
    elif modelVersion in ["CGM2.4","CGM2.4e"]:
        kv = plotViewers.LowerLimbMultiFootKinematicsPlotViewer(analysis,
                            pointLabelSuffix=pointLabelSuffix)
    else:
        raise Exception("[pyCGM2] Model version not known")

    kv.setConcretePlotFunction(plot.gaitDescriptivePlot)

    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    pf.setPath(DATA_PATH)
    pf.setPdfName(str(pdfFilename+"-descriptive Kinematics"))
    pf.plot()

    # filter 2 - consistency kinematic panel
    #-------------------------------------------
    # viewer
    if modelVersion in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e","CGM2.3","CGM2.3e"]:
        kv = plotViewers.LowerLimbKinematicsPlotViewer(analysis,
                                pointLabelSuffix=pointLabelSuffix)
    elif modelVersion in ["CGM2.4","CGM2.4e"]:
        kv = plotViewers.LowerLimbMultiFootKinematicsPlotViewer(analysis,
                                pointLabelSuffix=pointLabelSuffix)
    else:
        raise Exception("[pyCGM2] Model version not known")


    kv.setConcretePlotFunction(plot.gaitConsistencyPlot)

    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    pf.setPath(DATA_PATH)
    pf.setPdfName(str(pdfFilename+"-consistency Kinematics"))
    pf.plot()

    if kineticFlag:
        # filter 1 - descriptive kinematic panel
        #-------------------------------------------
        # viewer
        if modelVersion in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e","CGM2.3","CGM2.3e","CGM2.4","CGM2.4e"]:
            kv = plotViewers.LowerLimbKineticsPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix)
            kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
        if normativeDataset is not None:
            kv.setNormativeDataset(normativeDataset)

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.setPath(DATA_PATH)
        pf.setPdfName(str(pdfFilename+"-descriptive  Kinetics"))
        pf.plot()

        # filter 2 - consistency kinematic panel
        #-------------------------------------------
        # viewer
        if modelVersion in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e","CGM2.3","CGM2.3e","CGM2.4","CGM2.4e"]:
            kv = plotViewers.LowerLimbKineticsPlotViewer(analysis,
                                    pointLabelSuffix=pointLabelSuffix)
            kv.setConcretePlotFunction(plot.gaitConsistencyPlot)

        if normativeDataset is not None:
            kv.setNormativeDataset(normativeDataset)

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.setPath(DATA_PATH)
        pf.setPdfName(str(pdfFilename+"-consistency  Kinetics"))
        pf.plot()
