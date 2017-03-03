# -*- coding: utf-8 -*-
import sys
import pdb
import logging
import matplotlib.pyplot as plt
import argparse
import json
import os
from collections import OrderedDict

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
from pyCGM2.Tools import btkTools,nexusTools
    
if __name__ == "__main__":
   
    plt.close("all")
    DEBUG = False

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------INPUTS ------------------------------------

        if DEBUG:
            DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\CGM1-NexusPlugin\\CGM1-Calibration\\"
            calibrateFilenameLabelledNoExt = "static Cal 01-noKAD-noAnkleMed" #"static Cal 01-noKAD-noAnkleMed" #
            NEXUS.OpenTrial( str(DATA_PATH+calibrateFilenameLabelledNoExt), 30 )

        else:
            DATA_PATH, calibrateFilenameLabelledNoExt = NEXUS.GetTrialName()


        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ calibrateFilenameLabelled)


        # ----- Subject -----
        # need subject to find pycgm2 input files 
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.ckeckActivatedSubject(NEXUS,subjects,"LASI")
        logging.info(  "Subject name : " + subject  )

        # ---- pyCGM2 input files ----
        if not os.path.isfile( DATA_PATH + subject+"-pyCGM2.inputs"):
            raise Exception ("%s-pyCGM2.inputs file doesn't exist"%subject)
        else:
            inputs = json.loads(open(DATA_PATH +subject+'-pyCGM2.inputs').read(),object_pairs_hook=OrderedDict)

        # ---- configuration parameters ----
        pointSuffix = inputs["Global"]["Point suffix"]


        # -----infos--------     
        model = None if  inputs["Processing"]["Model"]=={} else inputs["Processing"]["Model"]  
        subject = None if inputs["Processing"]["Subject"]=={} else inputs["Processing"]["Subject"] 
        experimental = None if inputs["Processing"]["Experimental conditions"]=={} else inputs["Processing"]["Experimental conditions"] 

        # --------------------------PROCESSING --------------------------------
        # pycgm2-filter pipeline are gathered in a single function
        smartFunctions.staticProcessing_cgm1(calibrateFilenameLabelled, DATA_PATH,
                                             model,  subject, experimental,
                                             pointLabelSuffix = pointSuffix,
                                             exportSpreadSheet=False)


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")


