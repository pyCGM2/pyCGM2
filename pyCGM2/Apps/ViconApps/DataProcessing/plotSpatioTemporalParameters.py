# -*- coding: utf-8 -*-
"""Nexus Operation : **plotSpatioTemporalParameters**

The script displays spatio-temporal parameters (Velocity, cadence, duration of the gait phases...)

:param -ps, --pointSuffix [string]: suffix adds to the pyCGM2 nomenclature

Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>>  -ps=py

.. note::
    the spatio-temporal parameters are :
        * duration
        * cadence
        * stanceDuration
        * stancePhase
        * swingDuration
        * swingPhase
        * doubleStance1
        * doubleStance2
        * simpleStance
        * strideLength
        * stepLength
        * strideWidth
        * speed

.. warning::
    the spatio-temporal parameters are not stored in the c3d file yet.



"""
import traceback
import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt


# pyCGM2 settings
import pyCGM2


# vicon nexus
from viconnexusapi import ViconNexus

# pyCGM2 libraries
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Report import normativeDatasets

from pyCGM2.Nexus import  nexusTools,nexusFilters
from pyCGM2.Utils import files
from pyCGM2.Eclipse import eclipse

def main():

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM plot stp')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix added to pyCGM2 outputs')
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    ECLIPSE_MODE = False

    if not NEXUS_PYTHON_CONNECTED:
        raise Exception("Vicon Nexus is not running")

    pointSuffix = args.pointSuffix

    #--------------------------Data Location and subject-------------------------------------
    if eclipse.getCurrentMarkedNodes() is not None:
        LOGGER.logger.info("[pyCGM2] - Script worked with marked node of Vicon Eclipse")
        # --- acquisition file and path----
        DATA_PATH, modelledFilenames =eclipse.getCurrentMarkedNodes()
        ECLIPSE_MODE = True

    if not ECLIPSE_MODE:
        LOGGER.logger.info("[pyCGM2] - Script works with the loaded c3d in vicon Nexus")

        # --------------------------INPUTS ------------------------------------
        DATA_PATH, modelledFilenameNoExt = NEXUS.GetTrialName()

        modelledFilename = modelledFilenameNoExt+".c3d"

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.logger.info( "file: "+ modelledFilename)

    # ----- Subject -----
    # need subject to find input files
    # subjects = NEXUS.GetSubjectNames()
    subject = nexusTools.getActiveSubject(NEXUS)
    LOGGER.logger.info(  "Subject name : " + subject  )


    if not ECLIPSE_MODE:
        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,modelledFilenameNoExt,subject)
        acq = nacf.build()


        # --------------------------PROCESSING --------------------------------
        analysisInstance = analysis.makeAnalysis(DATA_PATH,[modelledFilename], pointLabelSuffix=pointSuffix,
                                                btkAcqs=[acq])

        outputName = modelledFilename
    else:
        # --------------------------PROCESSING --------------------------------
        analysisInstance = analysis.makeAnalysis(DATA_PATH,modelledFilenames, pointLabelSuffix=pointSuffix)
        outputName =  "Eclipse - SpatioTemporal parameters"


    plot.plot_spatioTemporal(DATA_PATH,analysisInstance,
        exportPdf=True,
        outputName=outputName)


if __name__ == "__main__":

    main()
