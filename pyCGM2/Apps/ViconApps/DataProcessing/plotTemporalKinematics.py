# -*- coding: utf-8 -*-
from pyCGM2.Nexus import nexusTools, nexusFilters
import warnings
import matplotlib.pyplot as plt
import argparse
from pyCGM2.Lib import plot
from viconnexusapi import ViconNexus
import pyCGM2
import pyCGM2
LOGGER = pyCGM2.LOGGER
warnings.simplefilter(action='ignore', category=FutureWarning)

# pyCGM2 settings

# vicon nexus

# pyCGM2 libraries


def main():
    """  Plot  temporal Kinematics from nexus-loaded trial

    Usage:

    ```bash
        python plotTemporalKinematics.py
        python plotTemporalKinematics.py  -ps CGM1
    ```

    Args:
        ['-ps','--pointSuffix'] (str): suffix added to model outputs ()

    """

    plt.close("all")

    parser = argparse.ArgumentParser(
        description='CGM plot Temporal Kinematics')
    parser.add_argument('-ps', '--pointSuffix', type=str,
                        help='suffix of model outputs')

    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED:

        pointSuffix = args.pointSuffix
        # --------------------------INPUTS ------------------------------------
        DATA_PATH, modelledFilenameNoExt = NEXUS.GetTrialName()

        modelledFilename = modelledFilenameNoExt+".c3d"

        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.logger.info("file: " + modelledFilename)

        # ----- Subject -----
        # need subject to find input files
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info("Subject name : " + subject)

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(
            DATA_PATH, modelledFilenameNoExt, subject)
        acq = nacf.build()

        # --------------------pyCGM2 MODEL ------------------------------
        plot.plotTemporalKinematic(DATA_PATH, modelledFilename, "LowerLimb",
                                   pointLabelSuffix=pointSuffix, exportPdf=True, btkAcq=acq)
        plot.plotTemporalKinematic(DATA_PATH, modelledFilename, "Trunk",
                                   pointLabelSuffix=pointSuffix, exportPdf=True, btkAcq=acq)
        plot.plotTemporalKinematic(DATA_PATH, modelledFilename, "UpperLimb",
                                   pointLabelSuffix=pointSuffix, exportPdf=True, btkAcq=acq)

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")


if __name__ == "__main__":

    main()
