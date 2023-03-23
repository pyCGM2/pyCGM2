# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/Plot
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--

from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusTools
import warnings
import matplotlib.pyplot as plt
import argparse
from pyCGM2.Utils import files
from pyCGM2.Lib import plot
from pyCGM2 import enums
from viconnexusapi import ViconNexus
import pyCGM2
import pyCGM2
LOGGER = pyCGM2.LOGGER
warnings.simplefilter(action='ignore', category=FutureWarning)

# pyCGM2 settings


# vicon nexus

# pyCGM2 libraries


def main():

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM plot temporal Kinetics')
    parser.add_argument('-ps', '--pointSuffix', type=str,
                        help='suffix of model outputs')

    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    if NEXUS_PYTHON_CONNECTED:
        args = parser.parse_args()

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
        plot.plotTemporalKinetic(DATA_PATH, modelledFilename, "LowerLimb",
                                 pointLabelSuffix=pointSuffix, exportPdf=True, btkAcq=acq)

    else:
        return parser


if __name__ == "__main__":

    main()
