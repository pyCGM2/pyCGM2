# -*- coding: utf-8 -*-
#import ipdb

from pyCGM2.Processing import c3dManager, cycle, analysis
from pyCGM2.Model.CGM2 import  cgm
from pyCGM2.Tools import btkTools
from pyCGM2.EMG import emgFilters
from pyCGM2 import enums

def makeAnalysis(type,modelVersion,DATA_PATH,
                    modelledFilenames,
                    subjectInfo, experimentalInfo,modelInfo,
                    pointLabelSuffix=""):


    #---- c3d manager
    c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
    cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
    cmf.enableEmg(False)
    trialManager = cmf.generate()


    #----cycles
    if type == "Gait":
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalTrials=trialManager.spatioTemporal["Trials"],
                                                   kinematicTrials = trialManager.kinematic["Trials"],
                                                   kineticTrials = trialManager.kinetic["Trials"],
                                                   emgTrials=trialManager.emg["Trials"])

    else:
        cycleBuilder = cycle.CyclesBuilder(spatioTemporalTrials=trialManager.spatioTemporal["Trials"],
                                                   kinematicTrials = trialManager.kinematic["Trials"],
                                                   kineticTrials = trialManager.kinetic["Trials"],
                                                   emgTrials=trialManager.emg["Trials"])

    cyclefilter = cycle.CyclesFilter()
    cyclefilter.setBuilder(cycleBuilder)
    cycles = cyclefilter.build()

    #----analysis
    if modelVersion=="CGM2.4":
        cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT["Left"].append("LForeFootAngles")
        cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT["Right"].append("RForeFootAngles")

    kinematicLabelsDict = cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT
    kineticLabelsDict = cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT



    if type == "Gait":
        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = kinematicLabelsDict,
                                                      kineticLabelsDict = kineticLabelsDict,
                                                      pointlabelSuffix = pointLabelSuffix)
    else:
        analysisBuilder = analysis.AnalysisBuilder(cycles,
                                                      kinematicLabelsDict = kinematicLabelsDict,
                                                      kineticLabelsDict = kineticLabelsDict,
                                                      pointlabelSuffix = pointLabelSuffix)

    finalmodelInfos = {"Version":modelVersion}
    if modelInfo is not None: finalmodelInfos.update(modelInfo)


    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setInfo(subject=subjectInfo, model=finalmodelInfos, experimental=experimentalInfo)
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()

    return analysisFilter.analysis

    #files.saveAnalysis(analysis,DATA_PATH,"Save_and_openAnalysis")

def processEMG(DATA_PATH, gaitTrials, EMG_LABELS, highPassFrequencies=[20,200],envelopFrequency=6.0, fileSuffix=""):


    for gaitTrial in gaitTrials:
        acq = btkTools.smartReader(DATA_PATH +gaitTrial)

        bf = emgFilters.BasicEmgProcessingFilter(acq,EMG_LABELS)
        bf.setHighPassFrequencies(highPassFrequencies[0],highPassFrequencies[1])
        bf.run()

        envf = emgFilters.EmgEnvelopProcessingFilter(acq,EMG_LABELS)
        envf.setCutoffFrequency(envelopFrequency)
        envf.run()

        outFilename = gaitTrial if fileSuffix=="" else gaitTrial+"_"+fileSuffix
        btkTools.smartWriter(acq,DATA_PATH+outFilename)



def makeEmgAnalysis(type,DATA_PATH,
                    processedEmgFiles,
                    emg_labels,
                    subjectInfo, experimentalInfo):


    c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,processedEmgFiles)
    cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
    cmf.enableSpatioTemporal(False)
    cmf.enableKinematic(False)
    cmf.enableKinetic(False)
    cmf.enableEmg(True)
    trialManager = cmf.generate()

    #---- GAIT CYCLES FILTER
    #--------------------------------------------------------------------------

    #----cycles
    if type == "Gait":
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalTrials=trialManager.spatioTemporal["Trials"],
                                                   kinematicTrials = trialManager.kinematic["Trials"],
                                                   kineticTrials = trialManager.kinetic["Trials"],
                                                   emgTrials=trialManager.emg["Trials"])

    else:
        cycleBuilder = cycle.CyclesBuilder(spatioTemporalTrials=trialManager.spatioTemporal["Trials"],
                                                   kinematicTrials = trialManager.kinematic["Trials"],
                                                   kineticTrials = trialManager.kinetic["Trials"],
                                                   emgTrials=trialManager.emg["Trials"])

    cyclefilter = cycle.CyclesFilter()
    cyclefilter.setBuilder(cycleBuilder)
    cycles = cyclefilter.build()

    emgLabelList  = [label+"_Rectify_Env" for label in emg_labels]

    if type == "Gait":
        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = None,
                                                      kineticLabelsDict = None,
                                                      emgLabelList = emgLabelList,
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=None,
                                                      experimentalInfos=experimentalInfo)
    else:
        analysisBuilder = analysis.AnalysisBuilder(cycles,
                                                      kinematicLabelsDict = None,
                                                      kineticLabelsDict = None,
                                                      emgLabelList = emgLabelList,
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=None,
                                                      experimentalInfos=experimentalInfo)

    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()

    analysisInstance = analysisFilter.analysis

    return analysisInstance

def normalizedEMG(analysis, EMG_LABELS,Contexts, method="MeanMax", fromOtherAnalysis=None):

    i=0
    for label in EMG_LABELS:


        envnf = emgFilters.EmgNormalisationProcessingFilter(analysis,label,Contexts[i])


        if fromOtherAnalysis is not None:
            envnf.setThresholdFromOtherAnalysis(fromOtherAnalysis)

        if method is not "MeanMax":
            envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MeanMax)
        elif method is not "MaxMax":
            envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MaxMax)
        elif method is not "MedianMax":
            envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MedianMax)

        envnf.run()
        i+=1
        del envnf
