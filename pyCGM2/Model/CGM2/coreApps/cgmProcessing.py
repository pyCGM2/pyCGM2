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
from pyCGM2.Processing import c3dManager
from pyCGM2.Processing.gaitAnalysis import smartFunctions
from pyCGM2.Model.CGM2 import  cgm,cgm2
from pyCGM2.Utils import files


def gaitprocessing(DATA_PATH, modelledFilename, modelVersion,
    modelInfo, subjectInfo, experimentalInfo,
    normativeData,
    pointSuffix,
    pdfname="gaitProcessing"):

    #---- c3d manager
    #--------------------------------------------------------------------------
    c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,[modelledFilename])
    cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
    cmf.enableEmg(False)
    trialManager = cmf.generate()

    #---- make analysis
    #-----------------------------------------------------------------------
            # pycgm2-filter pipeline are gathered in a single function
    if modelVersion in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e","CGM2.3","CGM2.3e"]:

        analysis = smartFunctions.make_analysis(trialManager,
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

    #---- plot panels
    #-----------------------------------------------------------------------
    smartFunctions.cgm_gaitPlots(modelVersion,analysis,trialManager.kineticFlag,
        DATA_PATH,pdfname,
        pointLabelSuffix=pointSuffix,
        normativeDataset=nds )

    plt.show()
