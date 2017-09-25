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


# openMA
import ma.io
import ma.body

# pyCGM2 libraries
from pyCGM2 import  smartFunctions
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import  nexusTools
from pyCGM2.Utils import files


if __name__ == "__main__":

    plt.close("all")
    DEBUG = False


    parser = argparse.ArgumentParser(description='CGM Static Processing')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------INPUTS ------------------------------------

        if DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.3\\c3dOnly\\"
            calibrateFilenameLabelledNoExt = "static" #"static Cal 01-noKAD-noAnkleMed" #
            NEXUS.OpenTrial( str(DATA_PATH+calibrateFilenameLabelledNoExt), 30 )

        else:
            DATA_PATH, calibrateFilenameLabelledNoExt = NEXUS.GetTrialName()


        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ calibrateFilenameLabelled)


        # ----- Subject -----
        # need subject to find pycgm2 input files
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

        # -----infos--------
        modelInfo = None if  infoSettings["Modelling"]["Model"]=={} else infoSettings["Modelling"]["Model"]
        subjectInfo = None if infoSettings["Processing"]["Subject"]=={} else infoSettings["Processing"]["Subject"]
        experimentalInfo = None if infoSettings["Processing"]["Experimental conditions"]=={} else infoSettings["Processing"]["Experimental conditions"]

        # --------------------------PROCESSING --------------------------------
        pdfFilename = calibrateFilenameLabelledNoExt
        smartFunctions.cgm_staticPlot(model,calibrateFilenameLabelled,
                                  DATA_PATH,pdfFilename,
                                pointLabelSuffix = pointSuffix)

        plt.show()
    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
