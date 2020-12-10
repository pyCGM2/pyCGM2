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

import logging
import argparse

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
from viconnexusapi import ViconNexus


# pyCGM2 libraries
from pyCGM2.Lib import eventDetector
from pyCGM2.Nexus import nexusTools,nexusFilters

def main():

    parser = argparse.ArgumentParser(description='DeepEventDetector')
    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # ----------------------INPUTS-------------------------------------------
        # --- acquisition file and path----
        DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ reconstructFilenameLabelled)

        #acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

        # --------------------------SUBJECT -----------------------------------

        # Notice : Work with ONE subject by session
        subject = nexusTools.getActiveSubject(NEXUS)
        logging.info(  "Subject name : " + subject  )

        # --- btk acquisition ----
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,reconstructFilenameLabelledNoExt,subject)
        acqGait = nacf.build()

        # ----------------------EVENT DETECTOR-------------------------------
        eventDetector.deepevent(acqGait)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusTools.createEvents(NEXUS,subject,acqGait,["Foot Strike","Foot Off"])
        # ========END of the nexus OPERATION if run from Nexus  =========



    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":


    main()