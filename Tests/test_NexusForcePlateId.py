# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_NexusForcePlateId.py::Tests
import pyCGM2; LOGGER = pyCGM2.LOGGER

import pyCGM2
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusTools


# vicon nexus
try:
    from viconnexusapi import ViconNexus
    NEXUS = ViconNexus.ViconNexus()
except:
    LOGGER.logger.warning("No Nexus connection")
else :
    class Tests:
        def test_0(self):

            NEXUS = ViconNexus.ViconNexus()

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\forcePlatesDetection\\"
            filename = "gait4FP"

            NEXUS.OpenTrial( str(DATA_PATH+filename), 30 )

            mfpa = nexusTools.getForcePlateAssignment(NEXUS)

            acqGait = btkTools.smartReader(str(DATA_PATH +  filename+ ".c3d"))
            mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
            mappedForcePlate1 = forceplates.matchingFootSideOnForceplate(acqGait,mfpa=mfpa)
