# -*- coding: utf-8 -*-
#import ipdb
import logging

from  pyCGM2 import enums
from pyCGM2.Processing import cycle,analysis,scores,exporter,c3dManager
from pyCGM2.Report import plot,plotFilters,plotViewers,normativeDatasets
from pyCGM2.Tools import trialTools

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
    cycleBuilder = cycle.CyclesBuilder(spatioTemporalTrials=trialManager.spatioTemporal["Trials"],
                                               kinematicTrials = trialManager.kinematic["Trials"],
                                               kineticTrials = trialManager.kinetic["Trials"],
                                               emgTrials=trialManager.emg["Trials"])



    cyclefilter = cycle.CyclesFilter()
    cyclefilter.setBuilder(cycleBuilder)
    cycles = cyclefilter.build()

    #---- GAIT ANALYSIS FILTER
    #--------------------------------------------------------------------------

    analysisBuilder = analysis.AnalysisBuilder(cycles,
                                                  kinematicLabelsDict = kinematicLabelsDict,
                                                  kineticLabelsDict = kineticLabelsDict,
                                                  pointlabelSuffix = pointLabelSuffix)

    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setInfo(subject=subjectInfo, model=modelInfo, experimental=experimentalInfo)
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()

    return analysisFilter.analysis
