# -*- coding: utf-8 -*-
"""Nexus Operation : **plotTemporalEmg**

The script displays rectified EMG with time as x-axis

:param -fso, --footStrikeOffset [int]: add an offset on all foot strike events
:param -foo, --footOffOffset [int]: add an offset on all foot off events


Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>>  -fso=10
    (add 10 frames to all foot strike events)


"""
import traceback
import os
import logging
import matplotlib.pyplot as plt
import argparse

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

#btk
from pyCGM2 import btk

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Events import events
from pyCGM2.Nexus import nexusTools

def main(args):

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # ----------------------INPUTS-------------------------------------------
        # --- acquisition file and path----
        DEBUG = False
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
        if args.footStrikeOffset is not None:
            evp.setFootStrikeOffset(args.footStrikeOffset)
        if args.footOffOffset is not None:
            evp.setFootOffOffset(args.footOffOffset)


        evf = events.EventFilter(evp,acqGait)
        evf.detect()

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusTools.createEvents(NEXUS,subject,acqGait,["Foot Strike","Foot Off"])
        # ========END of the nexus OPERATION if run from Nexus  =========

        if DEBUG:
            NEXUS.SaveTrial(30)


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ZeniDetector')
    parser.add_argument('-fso','--footStrikeOffset', type=int, help='systenatic foot strike offset on both side')
    parser.add_argument('-foo','--footOffOffset', type=int, help='systenatic foot off offset on both side')
    args = parser.parse_args()

    # ---- main script -----
    try:
        main(args)


    except Exception, errormsg:
        print "Error message: %s" % errormsg
        traceback.print_exc()
        print "Press return to exit.."
        raw_input()
