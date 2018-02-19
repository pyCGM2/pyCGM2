# -*- coding: utf-8 -*-
#import ipdb
import logging
import argparse
import matplotlib.pyplot as plt


# pyCGM2 settings
import pyCGM2

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Report import normativeDatasets,plot
from pyCGM2.Processing import c3dManager,exporter,scores
from pyCGM2.Processing.highLevel import standardSmartFunctions,gaitSmartFunctions
from pyCGM2.Model.CGM2 import  cgm,cgm2
from pyCGM2.Utils import files


def standardProcessing(DATA_PATH, modelledFilenames, modelVersion,
    modelInfo, subjectInfo, experimentalInfo,
    pointSuffix,
    outputPath=None,
    outputFilename="standardProcessing",
    exportXls=False):

    if outputPath is None:
        outputPath= DATA_PATH

    if isinstance(modelledFilenames,str):
        modelledFilenames = [modelledFilenames]

    #---- c3d manager
    #--------------------------------------------------------------------------
    c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
    cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
    cmf.enableEmg(False)
    trialManager = cmf.generate()

    #---- make analysis
    #-----------------------------------------------------------------------
            # pycgm2-filter pipeline are gathered in a single function
    if modelVersion in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e","CGM2.3","CGM2.3e","CGM2.4","CGM2.4e"]:

        if modelVersion in ["CGM2.4","CGM2.4e"]:
            cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT["Left"].append("LForeFoot")
            cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT["Right"].append("RForeFoot")

        analysis = standardSmartFunctions.make_analysis(trialManager,
                  cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
                  cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
                  modelInfo, subjectInfo, experimentalInfo,
                  pointLabelSuffix=pointSuffix)

    #---- export
    #-----------------------------------------------------------------------
    if exportXls:
        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(analysis)
        exportFilter.export(outputFilename, path=outputPath,excelFormat = "xls",mode="Advanced")

def gaitProcessing(DATA_PATH, modelledFilenames, modelVersion,
    modelInfo, subjectInfo, experimentalInfo,
    normativeData,
    pointSuffix,
    outputPath=None,
    outputFilename="gaitProcessing",
    exportXls=False,
    plot=True):

    if outputPath is None:
        outputPath= DATA_PATH


    if isinstance(modelledFilenames,str):
        modelledFilenames = [modelledFilenames]

    #---- c3d manager
    #--------------------------------------------------------------------------
    c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
    cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
    cmf.enableEmg(False)
    trialManager = cmf.generate()

    #---- make analysis
    #-----------------------------------------------------------------------
            # pycgm2-filter pipeline are gathered in a single function
    if modelVersion in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e","CGM2.3","CGM2.3e","CGM2.4","CGM2.4e"]:

        if modelVersion in ["CGM2.4","CGM2.4e"]:
            cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT["Left"].append("LForeFootAngles")
            cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT["Right"].append("RForeFootAngles")

        analysis = gaitSmartFunctions.make_analysis(trialManager,
              cgm.CGM1LowerLimbs.ANALYSIS_KINEMATIC_LABELS_DICT,
              cgm.CGM1LowerLimbs.ANALYSIS_KINETIC_LABELS_DICT,
              modelInfo, subjectInfo, experimentalInfo,
              pointLabelSuffix=pointSuffix)

    #---- normative dataset
    #-----------------------------------------------------------------------
    if normativeData["Author"] == "Schwartz2008":
        chosenModality = normativeData["Modality"]
        nds = normativeDatasets.Schwartz2008(chosenModality)    # modalites : "Very Slow" ,"Slow", "Free", "Fast", "Very Fast"
    elif normativeData["Author"] == "Pinzone2014":
        chosenModality = normativeData["Modality"]
        nds = normativeDatasets.Pinzone2014(chosenModality) # modalites : "Center One" ,"Center Two"

    #---- GPS
    gps =scores.CGM1_GPS(pointSuffix= pointSuffix)
    scf = scores.ScoreFilter(gps,analysis, nds)
    scf.compute()

    #---- export
    #-----------------------------------------------------------------------
    files.saveAnalysis(analysis,outputPath,outputFilename)

    if exportXls:
        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(analysis)
        exportFilter.export(outputFilename, path=outputPath,excelFormat = "xls",mode="Advanced")


    #---- plot panels
    #-----------------------------------------------------------------------
    if plot:
        gaitSmartFunctions.cgm_gaitPlots(modelVersion,analysis,trialManager.kineticFlag,
            outputPath,outputFilename,
            pointLabelSuffix=pointSuffix,
            normativeDataset=nds )

        plt.show()
