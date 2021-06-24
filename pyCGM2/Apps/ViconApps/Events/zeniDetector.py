# -*- coding: utf-8 -*-
import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse

# pyCGM2 settings
import pyCGM2

# vicon nexus
from viconnexusapi import ViconNexus


# pyCGM2 libraries
from pyCGM2.Lib import eventDetector
from pyCGM2.Nexus import nexusTools,nexusFilters

def main():

    parser = argparse.ArgumentParser(description='ZeniDetector')
    parser.add_argument('-fso','--footStrikeOffset', type=int, help='systenatic foot strike offset on both side')
    parser.add_argument('-foo','--footOffOffset', type=int, help='systenatic foot off offset on both side')
    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # ----------------------INPUTS-------------------------------------------
        # --- acquisition file and path----
        DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.logger.info( "calibration file: "+ reconstructFilenameLabelled)

        #acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

        # --------------------------SUBJECT -----------------------------------

        # Notice : Work with ONE subject by session
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info(  "Subject name : " + subject  )

        # --- btk acquisition ----
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,reconstructFilenameLabelledNoExt,subject)
        acqGait = nacf.build()


        # ----------------------EVENT DETECTOR-------------------------------
        footStrikeOffset = 0
        footOffOffset=0
        if args.footStrikeOffset is not None:
            footStrikeOffset = args.footStrikeOffset
        if args.footOffOffset is not None:
            footOffOffset = args.footOffOffset

        eventDetector.zeni(acqGait,footStrikeOffset=footStrikeOffset,footOffOffset=footOffOffset)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusTools.createEvents(NEXUS,subject,acqGait,["Foot Strike","Foot Off"])
        # ========END of the nexus OPERATION if run from Nexus  =========



    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":


    main()
