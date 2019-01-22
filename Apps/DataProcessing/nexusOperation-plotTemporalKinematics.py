# -*- coding: utf-8 -*-
import logging
import argparse
import matplotlib.pyplot as plt


# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
import pyCGM2
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Report import normativeDatasets

from pyCGM2.Nexus import  nexusTools
from pyCGM2.Utils import files

if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM Gait Processing')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')

    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED:

        #-----------------------SETTINGS---------------------------------------
        pointSuffix = args.pointSuffix if args.pointSuffix is not None else ""

        # --------------------------INPUTS ------------------------------------
        DEBUG= False
        if DEBUG:
            DATA_PATH = "C:\Users\HLS501\Documents\VICON DATA\pyCGM2-Data\Release Tests\CGM2.2\medial\\" #pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1\\native\\"
            modelledFilenameNoExt = "Gait Trial 01"# "gait trial" #"static Cal 01-noKAD-noAnkleMed" #
            NEXUS.OpenTrial( str(DATA_PATH+modelledFilenameNoExt), 30 )
        else:
            DATA_PATH, modelledFilenameNoExt = NEXUS.GetTrialName()


        modelledFilename = modelledFilenameNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "file: "+ modelledFilename)


        plotTemporalKinematic(DATA_PATH, modelledFilename,pointLabelSuffix=pointSuffix,exportPdf=True)



    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
