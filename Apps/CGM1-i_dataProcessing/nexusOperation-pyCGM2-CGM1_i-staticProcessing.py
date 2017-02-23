# -*- coding: utf-8 -*-
import sys
import pdb
import logging
import matplotlib.pyplot as plt
import argparse
import json
import os

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

         # ---- pyCGM2 files ----
        if not os.path.isfile( DATA_PATH + "pyCGM2.inputs"):
            raise Exception ("pyCGM2.inputs file doesn't exist")
        else:
            inputs = json.loads(open(DATA_PATH +'pyCGM2.inputs').read())

        # ---- configuration parameters ----
        pointSuffix = inputs["Calibration"]["Point suffix"]


        # -----infos--------     
        model = None if  inputs["Model"]=={} else inputs["Model"]  
        subject = None if inputs["Subject"]=={} else inputs["Subject"] 
        experimental = None if inputs["Experimental conditions"]=={} else inputs["Experimental conditions"] 

        # --------------------------PROCESSING --------------------------------
        # pycgm2-filter pipeline are gathered in a single function
        smartFunctions.staticProcessing_cgm1(calibrateFilenameLabelled, DATA_PATH,
                                             model,  subject, experimental,
                                             pointLabelSuffix = pointSuffix,
                                             exportSpreadSheet=False)


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")


