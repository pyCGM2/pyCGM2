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
    def croppedC3d(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"
        filenameNoExt = "gait_cropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        acqConstructorFilter = nexusFilters.NexusConstructAcquisitionFilter(filenameNoExt,subject)
        acq = acqConstructorFilter.build()

        acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

        np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
        np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fx1").GetValues(),acq0.GetAnalog("Force.Fx1").GetValues(),decimal=2)

        btkTools.smartWriter(acq,"gait_cropped_checked.c3d")

    @classmethod
    def croppedC3d_noX2d(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"
        filenameNoExt = "gait_cropped_nox2d"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        acqConstructorFilter = nexusFilters.NexusConstructAcquisitionFilter(filenameNoExt,subject)
        acq = acqConstructorFilter.build()

        acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

        np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
        np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fx1").GetValues(),acq0.GetAnalog("Force.Fx1").GetValues(),decimal=2)


        btkTools.smartWriter(acq,"gait_cropped - nox2d_checked.c3d")



    @classmethod
    def noCroppedC3d(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"
        filenameNoExt = "gait_noCropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        acqConstructorFilter = nexusFilters.NexusConstructAcquisitionFilter(filenameNoExt,subject)
        acq = acqConstructorFilter.build()

        acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

        np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
        np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fx1").GetValues(),acq0.GetAnalog("Force.Fx1").GetValues(),decimal=2)


        btkTools.smartWriter(acq,"gait_noCropped_checked.c3d")





    @classmethod
    def Kistler4_Noraxon1_Xsens1(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"
        filenameNoExt = "gait_cropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        acqConstructorFilter = nexusFilters.NexusConstructAcquisitionFilter(filenameNoExt,subject)
        acq = acqConstructorFilter.build()


        acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))


        np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
        np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fx1").GetValues(),acq0.GetAnalog("Force.Fx1").GetValues(),decimal=2)



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

        #btkTools.smartWriter(acq,"Kistler4_Noraxon1_Xsens1.c3d")


    @classmethod
    def Kistler4_Noraxon1_Xsens1_wrenchOuputs(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"
        filenameNoExt = "gait_cropped"

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

        np.testing.assert_array_almost_equal(grwc.GetItem(0).GetForce().GetValues(),grwc0.GetItem(0).GetForce().GetValues(),decimal=1)
        np.testing.assert_array_almost_equal(grwc.GetItem(0).GetMoment().GetValues(),grwc0.GetItem(0).GetMoment().GetValues(),decimal=1)
        np.testing.assert_array_almost_equal(grwc.GetItem(0).GetPosition().GetValues(),grwc0.GetItem(0).GetPosition().GetValues(),decimal=1)



        # plt.figure()
        # plt.plot(grwc.GetItem(0).GetForce().GetValues())
        # plt.plot(grwc0.GetItem(0).GetForce().GetValues(),"o")
        #
        #
        # plt.figure()
        # plt.plot(grwc.GetItem(0).GetMoment().GetValues())
        # plt.plot(grwc0.GetItem(0).GetMoment().GetValues(),"o")
        #
        #
        # plt.figure()
        # plt.plot(grwc.GetItem(0).GetPosition().GetValues())
        # plt.plot(grwc0.GetItem(0).GetPosition().GetValues(),"o")
        # plt.show()





if __name__ == "__main__":

    Tests.croppedC3d()
    Tests.noCroppedC3d()
    Tests.croppedC3d_noX2d()

    Tests.Kistler4_Noraxon1_Xsens1()
    Tests.Kistler4_Noraxon1_Xsens1_wrenchOuputs()
