# -*- coding: utf-8 -*-
import sys
import cPickle
import ipdb
import logging
import matplotlib.pyplot as plt
import argparse
import json
import os
from collections import OrderedDict
from shutil import copyfile


# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2 import  smartFunctions
from pyCGM2.Tools import btkTools
from pyCGM2.Report import normativeDatasets,plot
from pyCGM2.Processing import c3dManager
from pyCGM2.Model.CGM2 import  cgm,cgm2
from pyCGM2.Nexus import  nexusTools
from pyCGM2.Utils import files

if __name__ == "__main__":

    plt.close("all")
    DEBUG = False

    parser = argparse.ArgumentParser(description='CGM Gait Processing')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------INPUTS ------------------------------------

        if DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\CGM1-NexusPlugin\\pyCGM2- CGM1-KAD\\"
            modelledFilenameNoExt = "Gait Trial 02" #"static Cal 01-noKAD-noAnkleMed" #
            NEXUS.OpenTrial( str(DATA_PATH+modelledFilenameNoExt), 30 )


        else:
            DATA_PATH, modelledFilenameNoExt = NEXUS.GetTrialName()


        modelledFilename = modelledFilenameNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "file: "+ modelledFilename)

        # ----- Subject -----
        # need subject to find input files
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)

        # ---- pyCGM2 input files ----
        # info file
        infoSettings = files.manage_pycgm2SessionInfos(DATA_PATH,subject)

        # ---- configuration parameters ----
        if args.pointSuffix is not None:
            pointSuffix = args.pointSuffix
        else:
            pointSuffix = ""

        normativeData = infoSettings["Processing"]["Normative data"]


        # -----infos--------
        modelInfo = None if  infoSettings["Modelling"]["Model"]=={} else infoSettings["Modelling"]["Model"]
        subjectInfo = None if infoSettings["Processing"]["Subject"]=={} else infoSettings["Processing"]["Subject"]
        experimentalInfo = None if infoSettings["Processing"]["Experimental conditions"]=={} else infoSettings["Processing"]["Experimental conditions"]

        # --------------------------PROCESSING --------------------------------

        #---- c3d manager
        #--------------------------------------------------------------------------
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,[modelledFilename])
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()

        #---- make analysis
        #-----------------------------------------------------------------------
                # pycgm2-filter pipeline are gathered in a single function
        if model.version in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e","CGM2.3","CGM2.3e"]:

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
        smartFunctions.cgm_gaitPlots(model,analysis,trialManager.kineticFlag,
            DATA_PATH,modelledFilenameNoExt,
            pointLabelSuffix=pointSuffix,
            normativeDataset=nds )

        plt.show()
    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
