# -*- coding: utf-8 -*-
import ipdb
import logging
import numpy as np
import matplotlib.pyplot as plt
# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2 import btk
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusFilters




class Tests():

    @classmethod
    def Kistler4_Noraxon1_Xsens1(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"
        filenameNoExt = "gait"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        acqConstructorFilter = nexusFilters.NexusConstructAcquisitionFilter(filenameNoExt,subject)
        acq = acqConstructorFilter.build()


        acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

        np.testing.assert_equal(acq.GetPointFrequency(),acq0.GetPointFrequency())
        np.testing.assert_equal(acq.GetNumberAnalogSamplePerFrame(),acq0.GetNumberAnalogSamplePerFrame())
        np.testing.assert_equal(acq.GetAnalogFrequency(),acq0.GetAnalogFrequency())
        np.testing.assert_equal(btkTools.smartGetMetadata(acq,"FORCE_PLATFORM","USED"),btkTools.smartGetMetadata(acq0,"FORCE_PLATFORM","USED"))

        np.testing.assert_array_almost_equal(map(float,btkTools.smartGetMetadata(acq,"FORCE_PLATFORM","CORNERS")),
                                             map(float,btkTools.smartGetMetadata(acq0,"FORCE_PLATFORM","CORNERS")),decimal=2)
        np.testing.assert_array_almost_equal(map(float,btkTools.smartGetMetadata(acq,"FORCE_PLATFORM","ORIGIN")),
                                             map(float,btkTools.smartGetMetadata(acq0,"FORCE_PLATFORM","ORIGIN")),decimal=2)
        np.testing.assert_equal(map(float,btkTools.smartGetMetadata(acq,"FORCE_PLATFORM","CHANNEL")),
                                map(float,btkTools.smartGetMetadata(acq0,"FORCE_PLATFORM","CHANNEL")))

        #btkTools.smartWriter(acq,"sample0_checked.c3d")


    @classmethod
    def Kistler4_Noraxon1_Xsens1_wrenchOuputs(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"
        filenameNoExt = "gait"

        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]


        acqConstructorFilter = nexusFilters.NexusConstructAcquisitionFilter(filenameNoExt,subject)
        acq = acqConstructorFilter.build()

        #btkTools.smartWriter(acq,"NEWC3D.c3d")

        # --- ground reaction force wrench ---
        pfe = btk.btkForcePlatformsExtractor()
        pfe.SetInput(acq)
        pfc = pfe.GetOutput()

        grwf = btk.btkGroundReactionWrenchFilter()
        grwf.SetInput(pfc)
        grwc = grwf.GetOutput()
        grwc.Update()


        # --- reference values ---
        acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))
        pfe0 = btk.btkForcePlatformsExtractor()
        pfe0.SetInput(acq0)
        pfc0 = pfe0.GetOutput()

        grwf0 = btk.btkGroundReactionWrenchFilter()
        grwf0.SetInput(pfc0)
        grwc0 = grwf0.GetOutput()
        grwc0.Update()

        aff = (acq0.GetFirstFrame()-1)*acq0.GetNumberAnalogSamplePerFrame()
        alf = (acq0.GetLastFrame())*acq0.GetNumberAnalogSamplePerFrame()


        plt.figure()
        plt.plot(grwc.GetItem(0).GetForce().GetValues()[aff:alf])
        plt.plot(grwc0.GetItem(0).GetForce().GetValues(),"o")


        plt.figure()
        plt.plot(grwc.GetItem(0).GetMoment().GetValues()[aff:alf])
        plt.plot(grwc0.GetItem(0).GetMoment().GetValues(),"o")


        plt.figure()
        plt.plot(grwc.GetItem(0).GetPosition().GetValues()[aff:alf])
        plt.plot(grwc0.GetItem(0).GetPosition().GetValues(),"o")
        plt.show()



if __name__ == "__main__":

    Tests.Kistler4_Noraxon1_Xsens1()
    Tests.Kistler4_Noraxon1_Xsens1_wrenchOuputs()
