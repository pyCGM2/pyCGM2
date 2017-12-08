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
from pyCGM2.Processing.gaitAnalysis import smartFunctions
from pyCGM2.Nexus import  nexusTools
from pyCGM2.Utils import files


if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM Static Processing')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--DEBUG', action='store_true', help='debug model. load file into nexus externally')
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED: # run Operation

        #-----------------------SETTINGS---------------------------------------
        pointSuffix = args.pointSuffix if args.pointSuffix is not None else ""

        # --------------------------INPUTS ------------------------------------
        if args.DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\CGM1\\native\\"
            calibrateFilenameLabelledNoExt = "static"
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


        ## --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)
        modelVersion = model.version

        # --------------------SESSION INFOS ------------------------------
        # -----infos--------
        modelInfo = None #if  infoSettings["Modelling"]["Model"]=={} else infoSettings["Modelling"]["Model"]
        subjectInfo = None #if infoSettings["Processing"]["Subject"]=={} else infoSettings["Processing"]["Subject"]
        experimentalInfo = None #if infoSettings["Processing"]["Experimental conditions"]=={} else infoSettings["Processing"]["Experimental conditions"]

        # --------------------------PROCESSING --------------------------------
        # call processing.gaitAnalysis.processing directly
        smartFunctions.cgm_staticPlot(modelVersion,calibrateFilenameLabelled,
                                  DATA_PATH,
                                  pdfFilename = calibrateFilenameLabelledNoExt,
                                  pointLabelSuffix = pointSuffix)

        plt.show()
    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
