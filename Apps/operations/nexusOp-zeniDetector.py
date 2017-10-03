
# -*- coding: utf-8 -*-
import os
import logging
import matplotlib.pyplot as plt
import argparse

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

#btk
import btk

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Events import events
from pyCGM2.Nexus import nexusTools

if __name__ == "__main__":


    DEBUG = False

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    parser = argparse.ArgumentParser(description='ZeniDetector')
    args = parser.parse_args()

    if NEXUS_PYTHON_CONNECTED: # run Operation

        # ----------------------INPUTS-------------------------------------------
        # --- acquisition file and path----
        if DEBUG:
            DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\Salford\\Alana MoCap data\\MRI-US-01 - myProcess\\PIG\\"
            reconstructFilenameLabelledNoExt = "MRI-US-01, 2008-08-08, 3DGA 16" #"static Cal 01-noKAD-noAnkleMed" #

            NEXUS.OpenTrial( str(DATA_PATH+reconstructFilenameLabelledNoExt), 10 )

        else:
            DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ reconstructFilenameLabelled)

        # --- btk acquisition ----
        acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))


        #   check if acq was saved with only one  activated subject
        if acqGait.GetPoint(0).GetLabel().count(":"):
            raise Exception("[pyCGM2] Your Trial c3d was saved with two activate subject. Re-save it with only one before pyCGM2 calculation")

        # --------------------------SUBJECT -----------------------------------

        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )

        # ----------------------EVENT DETECTOR-------------------------------
        evp = events.ZeniProcedure()
        evf = events.EventFilter(evp,acqGait)
        evf.detect()

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusTools.createEvents(NEXUS,subject,acqGait,["Foot Strike","Foot Off"])
        # ========END of the nexus OPERATION if run from Nexus  =========

        if DEBUG:
            NEXUS.SaveTrial(30)


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
