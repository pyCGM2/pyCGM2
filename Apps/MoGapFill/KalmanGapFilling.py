# -*- coding: utf-8 -*-
"""Nexus Operation : **KalmanGapFilling**

Low dimensional Kalman smoother that fills gaps in motion capture marker trajectories

This repository is a  Python implementation of a gap filling algorithm
(http://dx.doi.org/10.1016/j.jbiomech.2016.04.016)
that smooths trajectories in low dimensional subspaces, together with a Python plugin for Vicon Nexus.
"""


import sys
import numpy as np
import traceback
import logging

import pyCGM2
import ViconNexus

from pyCGM2.Gap import gapFilling
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusTools



def main():

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED: # run Operation

        DATA_PATH, filenameLabelledNoExt = NEXUS.GetTrialName()

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "file: "+ filenameLabelledNoExt)

        subject = NEXUS.GetSubjectNames()[0]
        print "Gap filling for subject ", subject

        acq = btkTools.smartReader(str(DATA_PATH+filenameLabelledNoExt+".c3d"))

        gfp =  gapFilling.LowDimensionalKalmanFilterProcedure()
        gff = gapFilling.GapFillingFilter(gfp,acq)
        gff.fill()

        filledAcq  = gff.getFilledAcq()
        filledMarkers  = gff.getFilledMarkers()

        for marker in filledMarkers:
            nexusTools.setTrajectoryFromAcq(NEXUS,subject,marker,filledAcq)

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":


    # ---- main script -----
    try:
        main()


    except Exception, errormsg:
        print "Error message: %s" % errormsg
        traceback.print_exc()
        raise
