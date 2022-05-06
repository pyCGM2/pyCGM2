# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/Gap Filling
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--
from pyCGM2.Nexus import nexusTools, nexusFilters
from pyCGM2.Gap import gapFilters, gapFillingProcedures
from viconnexusapi import ViconNexus
import pyCGM2
LOGGER = pyCGM2.LOGGER


def main():
    """  Run Kalman gap filling method on the  nexus-loaded trial

    Usage:

    ```bash
        python KalmanGapFilling.py
    ```

    """

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

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
        raise Exception("NO Nexus connection. Turn on Nexus")


if __name__ == "__main__":

    main()
