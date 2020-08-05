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
import logging
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt


# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Report import normativeDatasets

from pyCGM2.Nexus import  nexusTools,nexusFilters
from pyCGM2.Utils import files

def main():

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM plot stp')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix added to pyCGM2 outputs')
    args = parser.parse_args()



    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED:

        pointSuffix = args.pointSuffix
        # --------------------------INPUTS ------------------------------------
        DATA_PATH, modelledFilenameNoExt = NEXUS.GetTrialName()


        modelledFilename = modelledFilenameNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "file: "+ modelledFilename)

        # ----- Subject -----
        # need subject to find input files
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)
        modelVersion = model.version

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,modelledFilenameNoExt,subject)
        acq = nacf.build()


        # --------------------------PROCESSING --------------------------------
        analysisInstance = analysis.makeAnalysis(DATA_PATH,[modelledFilename], pointLabelSuffix=pointSuffix,
                                                btkAcqs=[acq])

        plot.plot_spatioTemporal(DATA_PATH,analysisInstance,
            exportPdf=True,
            outputName=modelledFilename)


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":

    main()
