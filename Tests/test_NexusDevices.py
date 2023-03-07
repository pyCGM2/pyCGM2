# coding: utf-8
# pytest -s --disable-pytest-warnings  test_NexusDevices.py::Tests
import pyCGM2; LOGGER = pyCGM2.LOGGER
import numpy as np

# pyCGM2 settings
import pyCGM2
from pyCGM2.Tools import btkTools

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")


try:
    from viconnexusapi import ViconNexus
    NEXUS = ViconNexus.ViconNexus()
except:
    LOGGER.logger.warning("No Nexus connection")
else :

    from pyCGM2.Nexus import nexusTools,Devices


    class Tests:
        def test_analogDeviceTest(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\c3d_x2d\\"

            filenameNoExt = "gait_GAP"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )

            framerate = NEXUS.GetFrameRate()
            frames = NEXUS.GetFrameCount()
            analogFrameRate = NEXUS.GetDeviceDetails(1)[2]

            device = Devices.AnalogDevice(6)
            output = device.getChannels()


        def test_forcePlateTest(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\c3d_x2d\\"

            filenameNoExt = "gait_GAP"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )

            #----Btk----
            acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

            analogFx = acq0.GetAnalog("Force.Fx1").GetValues()
            analogFy = acq0.GetAnalog("Force.Fy1").GetValues()
            analogFz = acq0.GetAnalog("Force.Fz1").GetValues()
            analogMx = acq0.GetAnalog("Moment.Mx1").GetValues()
            analogMy = acq0.GetAnalog("Moment.My1").GetValues()
            analogMz = acq0.GetAnalog("Moment.Mz1").GetValues()



            pfe = btk.btkForcePlatformsExtractor()
            pfe.SetInput(acq0)
            pfc = pfe.GetOutput()
            pfc.Update()
            btkfp0 = pfc.GetItem(0)
            ch0_Fx = btkfp0.GetChannel(0).GetValues()
            ch1_Fy = btkfp0.GetChannel(1).GetValues()
            ch2_Fz = btkfp0.GetChannel(2).GetValues()
            ch3_Mx = btkfp0.GetChannel(3).GetValues()
            ch4_My = btkfp0.GetChannel(4).GetValues()
            ch5_Mz = btkfp0.GetChannel(5).GetValues()


            # --- ground reaction force wrench ---
            grwf = btk.btkGroundReactionWrenchFilter()
            grwf.SetInput(pfc)
            grwc = grwf.GetOutput()
            grwc.Update()

            grw0_force = grwc.GetItem(0).GetForce().GetValues()

            #----Nexus----
            nexusForcePlate = Devices.ForcePlate(1)
            forceLocal = nexusForcePlate.getLocalReactionForce()
            momentLocal = nexusForcePlate.getLocalReactionMoment()
