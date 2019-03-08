# -*- coding: utf-8 -*-
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2.ForcePlates import forceplates
import matplotlib.pyplot as plt
# vicon nexus
import ViconNexus


# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusTools




if __name__ == "__main__":

    NEXUS = ViconNexus.ViconNexus()
    DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\Nexus Operations\\Session1\\"
    filename = "100Hz_All_01"
    
    DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\Nexus Operations\\forcePlatesDetection\\"
    filename = "gait4FP"


    NEXUS.OpenTrial( str(DATA_PATH+filename), 30 )

    mfpa = nexusTools.getForcePlateAssignment(NEXUS)

    acqGait = btkTools.smartReader(str(DATA_PATH +  filename+ ".c3d"))
    mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
    mappedForcePlate1 = forceplates.matchingFootSideOnForceplate(acqGait,mfpa=mfpa)

    print mfpa
    print mappedForcePlate
    print mappedForcePlate1
