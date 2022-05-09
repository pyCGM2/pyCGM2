# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/Gap Filling
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--
import argparse
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusTools
from pyCGM2.Gap import gapFillingProcedures
from pyCGM2.Gap import gapFilters
from viconnexusapi import ViconNexus
import pyCGM2
LOGGER = pyCGM2.LOGGER


def main():
    parser = argparse.ArgumentParser(description='Zeni kinematic-based gait event Detector')


    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    if NEXUS_PYTHON_CONNECTED:  # run Operation

        DATA_PATH, filenameLabelledNoExt = NEXUS.GetTrialName()

        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.logger.info("file: " + filenameLabelledNoExt)

        # checkActivatedSubject(NEXUS,subjects)
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info("Gap filling for subject %s" % (subject))

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(
            DATA_PATH, filenameLabelledNoExt, subject)
        acq = nacf.build()

        #acq = btkTools.smartReader(str(DATA_PATH+filenameLabelledNoExt+".c3d"))

        gfp = gapFillingProcedures.LowDimensionalKalmanFilterProcedure()
        gff = gapFilters.GapFillingFilter(gfp, acq)
        gff.fill()

        filledAcq = gff.getFilledAcq()
        filledMarkers = gff.getFilledMarkers()

        for marker in filledMarkers:
            nexusTools.setTrajectoryFromAcq(NEXUS, subject, marker, filledAcq)

    else:
        return parser


if __name__ == "__main__":

    main()
