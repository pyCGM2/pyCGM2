# -*- coding: utf-8 -*-
import logging
import argparse
import matplotlib.pyplot as plt


# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries

from pyCGM2.Model.CGM2.coreApps import cgmProcessing
from pyCGM2.Nexus import  nexusTools
from pyCGM2.Utils import files

if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM Gait Processing')
    parser.add_argument('--DEBUG', action='store_true', help='debug model. load file into nexus externally')
    parser.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
    parser.add_argument('-ndm','--normativeDataModality', type=str,
                        help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                        default="Free")
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED:

        #-----------------------SETTINGS---------------------------------------
        pointSuffix = args.pointSuffix if args.pointSuffix is not None else ""
        normativeData = {"Author" : args.normativeData, "Modality" : args.normativeDataModality}

        # --------------------------INPUTS ------------------------------------
        if args.DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\CGM1\\native\\"
            modelledFilenameNoExt = "gait trial" #"static Cal 01-noKAD-noAnkleMed" #
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
        modelVersion = model.version

        # --------------------SESSION INFOS ------------------------------
        # -----infos--------
        modelInfo = None #if  infoSettings["Modelling"]["Model"]=={} else infoSettings["Modelling"]["Model"]
        subjectInfo = None #if infoSettings["Processing"]["Subject"]=={} else infoSettings["Processing"]["Subject"]
        experimentalInfo = None #if infoSettings["Processing"]["Experimental conditions"]=={} else infoSettings["Processing"]["Experimental conditions"]

        # --------------------------PROCESSING --------------------------------

        cgmProcessing.gaitprocessing(DATA_PATH,modelledFilename,modelVersion,
            modelInfo, subjectInfo, experimentalInfo,
            normativeData,
            pointSuffix,
            pdfname = modelledFilenameNoExt)

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
