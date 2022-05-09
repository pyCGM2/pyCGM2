# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/Events
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusTools
from pyCGM2.Lib import eventDetector
from viconnexusapi import ViconNexus
import argparse
import pyCGM2
LOGGER = pyCGM2.LOGGER


def main():

    parser = argparse.ArgumentParser(description='Zeni kinematic-based gait event Detector')
    parser.add_argument('-fso', '--footStrikeOffset', type=int,
                        help='systenatic foot strike offset on both side')
    parser.add_argument('-foo', '--footOffOffset', type=int,
                        help='systenatic foot off offset on both side')
    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False


    if NEXUS_PYTHON_CONNECTED: # run Operation
        args = parser.parse_args()

        # ----------------------INPUTS-------------------------------------------
        # --- acquisition file and path----
        DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.logger.info("calibration file: " + reconstructFilenameLabelled)

        #acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

        # --------------------------SUBJECT -----------------------------------

        # Notice : Work with ONE subject by session
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info("Subject name : " + subject)

        # --- btk acquisition ----
        nacf = nexusFilters.NexusConstructAcquisitionFilter(
            DATA_PATH, reconstructFilenameLabelledNoExt, subject)
        acqGait = nacf.build()

        # ----------------------EVENT DETECTOR-------------------------------
        footStrikeOffset = 0
        footOffOffset = 0
        if args.footStrikeOffset is not None:
            footStrikeOffset = args.footStrikeOffset
        if args.footOffOffset is not None:
            footOffOffset = args.footOffOffset

        eventDetector.zeni(
            acqGait, footStrikeOffset=footStrikeOffset, footOffOffset=footOffOffset)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusTools.createEvents(NEXUS, subject, acqGait, [
                                "Foot Strike", "Foot Off"])
        # ========END of the nexus OPERATION if run from Nexus  =========

    else:
        return parser


if __name__ == "__main__":

    main()
