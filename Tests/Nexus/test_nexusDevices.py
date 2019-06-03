# -*- coding: utf-8 -*-
import ipdb
import logging
import numpy as np
import matplotlib.pyplot as plt
# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

from pyCGM2 import btk
# vicon nexus
import ViconNexus


# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusTools,Devices



class Tests():


    @classmethod
    def analogDeviceTest(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\NexusAPI\\c3d_x2d\\"

        filenameNoExt = "gait_GAP"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )

        framerate = NEXUS.GetFrameRate()
        frames = NEXUS.GetFrameCount()
        analogFrameRate = NEXUS.GetDeviceDetails(1)[2]


        device = Devices.AnalogDevice(6)
        output = device.getChannels()

        #ipdb.set_trace()


    @classmethod
    def forcePlateTest(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\NexusAPI\\c3d_x2d\\"

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

        # ch3_Mx = btkfp0.GetChannel(3).GetValues()
        # ch4_My = btkfp0.GetChannel(4).GetValues()
        # ch5_Mz = btkfp0.GetChannel(5).GetValues()



        #----Nexus----
        nexusForcePlate = Devices.ForcePlate(1)
        forceLocal = nexusForcePlate.getLocalReactionForce()


        # plt.figure()
        # plt.plot(forceLocal[2310:,0],"-r")
        # plt.plot(ch0_Fx,"or")
        # plt.plot(analogFx,"oc")
        # #plt.plot(grw0_force[:,0],"-g")
        # #plt.plot(forceGlobal[2320:,0],"og")
        #
        #
        # plt.figure()
        # plt.plot(forceLocal[2310:,1],"-r")
        # plt.plot(ch1_Fy,"or")
        # plt.plot(analogFy,"oc")
        #
        #
        # plt.figure()
        # plt.plot(forceLocal[2310:,2],"-r")
        # plt.plot(ch2_Fz,"or")
        # plt.plot(analogFz,"oc")

        # plt.show()



        momentLocal = nexusForcePlate.getLocalReactionMoment()

        # plt.figure()
        # plt.plot(momentLocal[2310:,0],"-r")
        # plt.plot(ch3_Mx,"or")
        # plt.plot(analogMx,"oc")
        #plt.plot(grw0_force[:,0],"-g")
        #plt.plot(forceGlobal[2320:,0],"og")


        # plt.figure()
        # plt.plot(momentLocal[2310:,1],"-r")
        # plt.plot(ch4_My,"or")
        # plt.plot(analogMy,"oc")
        #
        #
        # plt.figure()
        # plt.plot(momentLocal[2310:,2],"-r")
        # plt.plot(ch5_Mz,"or")
        # plt.plot(analogMz,"oc")
        #
        # plt.show()



        #import ipdb; ipdb.set_trace()



if __name__ == "__main__":

    Tests.analogDeviceTest()
    Tests.forcePlateTest()
