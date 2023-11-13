# coding: utf-8
# pytest -s --disable-pytest-warnings  test_NexusBtkAcqFilter.py::Tests::test_croppedC3d
# from __future__ import unicode_literals
import pyCGM2; LOGGER = pyCGM2.LOGGER
import numpy as np

import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusTools
try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")


from viconnexusapi import ViconNexus
try:
    from viconnexusapi import ViconNexus
    NEXUS = ViconNexus.ViconNexus()
except:
    LOGGER.logger.warning("No Nexus connection")
else :

    class Tests:
        def test_croppedC3d(self):
            NEXUS = ViconNexus.ViconNexus()

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample0\\"
            filenameNoExt = "gait_cropped"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))


            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fx1").GetValues(),acq0.GetAnalog("Force.Fx1").GetValues(),decimal=2)


        def test_croppedC3d_noX2d(self):
            NEXUS = ViconNexus.ViconNexus()

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample0\\"
            filenameNoExt = "gait_cropped_nox2d"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            subject = nexusTools.getActiveSubject(NEXUS)

                    # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fx1").GetValues(),acq0.GetAnalog("Force.Fx1").GetValues(),decimal=2)

        def test_noCroppedC3d(self):
            NEXUS = ViconNexus.ViconNexus()



            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample0\\"
            filenameNoExt = "gait_noCropped"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )

            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fx1").GetValues(),acq0.GetAnalog("Force.Fx1").GetValues(),decimal=2)


        def test_Kistler4_Noraxon1_Xsens1(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample0\\"
            filenameNoExt = "gait_cropped"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()


            acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))


            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fx1").GetValues(),acq0.GetAnalog("Force.Fx1").GetValues(),decimal=2)



            np.testing.assert_equal(acq.GetPointFrequency(),acq0.GetPointFrequency())
            np.testing.assert_equal(acq.GetNumberAnalogSamplePerFrame(),acq0.GetNumberAnalogSamplePerFrame())
            np.testing.assert_equal(acq.GetAnalogFrequency(),acq0.GetAnalogFrequency())
            np.testing.assert_equal(btkTools.smartGetMetadata(acq,"FORCE_PLATFORM","USED"),btkTools.smartGetMetadata(acq0,"FORCE_PLATFORM","USED"))

            np.testing.assert_array_almost_equal([float(x) for x in np.asarray(btkTools.smartGetMetadata(acq,"FORCE_PLATFORM","CORNERS"))],
                                                [float(x) for x in np.asarray(btkTools.smartGetMetadata(acq0,"FORCE_PLATFORM","CORNERS"))],decimal=2)

            np.testing.assert_array_almost_equal([float(x) for x in np.asarray(btkTools.smartGetMetadata(acq,"FORCE_PLATFORM","ORIGIN"))],
                                                [float(x) for x in np.asarray(btkTools.smartGetMetadata(acq0,"FORCE_PLATFORM","ORIGIN"))],decimal=2)

            np.testing.assert_array_almost_equal([float(x) for x in np.asarray(btkTools.smartGetMetadata(acq,"FORCE_PLATFORM","CHANNEL"))],
                                                [float(x) for x in np.asarray(btkTools.smartGetMetadata(acq0,"FORCE_PLATFORM","CHANNEL"))],decimal=2)


        def test_Kistler4_Noraxon1_Xsens1_wrenchOuputs(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample0\\"
            filenameNoExt = "gait_cropped"

            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

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
            acq0 = btkTools.smartReader(DATA_PATH+ filenameNoExt+".c3d")
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

        def test_modelOutputs(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample0\\"
            filenameNoExt = "gait_cropped_ModelOutputDynamic"

            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

            np.testing.assert_array_almost_equal(acq.GetPoint("LAnkleAngles").GetValues(),acq0.GetPoint("LAnkleAngles").GetValues(),decimal=2)


        # def test_noX2d_manualCropping(self):
        #     NEXUS = ViconNexus.ViconNexus()
        #     NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
        #
        #     DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample3\\"
        #     filenameNoExt = "capture 01"
        #
        #     NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        #     NEXUS.SetTrialRegionOfInterest(350, 500)
        #     subject = nexusTools.getActiveSubject(NEXUS)
        #
        #     # btkAcq builder
        #     nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
        #     acq = nacf.build()
        #
        #     acq0 = btkTools.smartReader(str(DATA_PATH+ "Capture 01-cropped.c3d"))
        #
        #     np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
        #     #np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fx1").GetValues(),acq0.GetAnalog("Force.Fx1").GetValues(),decimal=2)

        def test_blueTrident(self):
            NEXUS = ViconNexus.ViconNexus()

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"LowLevel\\IMU\\BlueTrident-markers\\"
            filenameNoExt = "pycgm2-data01"

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"LowLevel\\IMU\\BlueTrident-markers\\"
            filenameNoExt = "pycgm2-data01"

            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            
            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartWriter(acq,"verif.c3d")

    class TestsX2d:

        def test_noCroppedC3d(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample_withx2d\\"
            filenameNoExt = "gait_noCropped"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )

            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz1").GetValues(),acq0.GetAnalog("Force.Fz1").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz2").GetValues(),acq0.GetAnalog("Force.Fz2").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz3").GetValues(),acq0.GetAnalog("Force.Fz3").GetValues(),decimal=2)

        def test_croppedC3d(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample_withx2d\\"
            filenameNoExt = "gait_cropped"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )

            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()


            acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz1").GetValues(),acq0.GetAnalog("Force.Fz1").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz2").GetValues(),acq0.GetAnalog("Force.Fz2").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz3").GetValues(),acq0.GetAnalog("Force.Fz3").GetValues(),decimal=2)

        def test_interactiveCropping_fromNoCropped(self):
            NEXUS = ViconNexus.ViconNexus()



            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample_withx2d\\"
            filenameNoExt = "gait_noCropped"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            NEXUS.SetTrialRegionOfInterest(300, 400)

            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartReader(str(DATA_PATH+ "forCheckingInteractiveCropped\\gait_Cropped - 300-400.c3d"))

            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz1").GetValues(),acq0.GetAnalog("Force.Fz1").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz2").GetValues(),acq0.GetAnalog("Force.Fz2").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz3").GetValues(),acq0.GetAnalog("Force.Fz3").GetValues(),decimal=2)

        def test_interactiveCropping_fromCropped(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample_withx2d\\"
            filenameNoExt = "gait_cropped"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            NEXUS.SetTrialRegionOfInterest(300, 400)

            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartReader(str(DATA_PATH+ "forCheckingInteractiveCropped\\gait_cropped - 300-400.c3d"))

            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz1").GetValues(),acq0.GetAnalog("Force.Fz1").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz2").GetValues(),acq0.GetAnalog("Force.Fz2").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz3").GetValues(),acq0.GetAnalog("Force.Fz3").GetValues(),decimal=2)



    class TestsNOX2d():

        def test_noCroppedC3d(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample_NOx2d\\"
            filenameNoExt = "gait_noCropped"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            # NEXUS.SetTrialRegionOfInterest(300, 400)

            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz1").GetValues(),acq0.GetAnalog("Force.Fz1").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz2").GetValues(),acq0.GetAnalog("Force.Fz2").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz3").GetValues(),acq0.GetAnalog("Force.Fz3").GetValues(),decimal=2)



        def test_croppedC3d(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample_NOx2d\\"
            filenameNoExt = "gait_cropped"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            # NEXUS.SetTrialRegionOfInterest(300, 400)

            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartReader(str(DATA_PATH+ filenameNoExt+".c3d"))

            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz1").GetValues(),acq0.GetAnalog("Force.Fz1").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz2").GetValues(),acq0.GetAnalog("Force.Fz2").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz3").GetValues(),acq0.GetAnalog("Force.Fz3").GetValues(),decimal=2)



        def test_interactiveCropping_fromNoCropped(self):
            NEXUS = ViconNexus.ViconNexus()



            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample_NOx2d\\"
            filenameNoExt = "gait_noCropped"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            NEXUS.SetTrialRegionOfInterest(300, 400)

            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartReader(str(DATA_PATH+ "forCheckingInteractiveCropped\\gait_Cropped - 300-400.c3d"))

            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz1").GetValues(),acq0.GetAnalog("Force.Fz1").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz2").GetValues(),acq0.GetAnalog("Force.Fz2").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz3").GetValues(),acq0.GetAnalog("Force.Fz3").GetValues(),decimal=2)

            btkTools.smartWriter(acq,"TestsNOX2d_interactiveCropping_fromNoCropped.c3d")



        def test_interactiveCropping_fromCropped(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"NexusAPI\\BtkAcquisitionCreator\\sample_NOx2d\\"
            filenameNoExt = "gait_cropped"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            NEXUS.SetTrialRegionOfInterest(300, 400)

            subject = nexusTools.getActiveSubject(NEXUS)

            # btkAcq builder
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,filenameNoExt,subject)
            acq = nacf.build()

            acq0 = btkTools.smartReader(str(DATA_PATH+ "forCheckingInteractiveCropped\\gait_cropped - 300-400.c3d"))

            np.testing.assert_array_almost_equal(acq.GetPoint("LTHI").GetValues(),acq0.GetPoint("LTHI").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz1").GetValues(),acq0.GetAnalog("Force.Fz1").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz2").GetValues(),acq0.GetAnalog("Force.Fz2").GetValues(),decimal=2)
            np.testing.assert_array_almost_equal(acq.GetAnalog("Force.Fz3").GetValues(),acq0.GetAnalog("Force.Fz3").GetValues(),decimal=2)

        