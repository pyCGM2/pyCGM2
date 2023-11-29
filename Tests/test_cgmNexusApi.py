# coding: utf-8
# pytest -s --disable-pytest-warnings  test_cgmNexusApi.py::Tests
import os
import pyCGM2

LOGGER = pyCGM2.LOGGER
import btk

from pyCGM2.Tools import btkTools
from pyCGM2.Utils import testingUtils

try:
    from viconnexusapi import ViconNexus
    NEXUS = ViconNexus.ViconNexus()
except:
    LOGGER.logger.warning("No Nexus connection")
else :
    
    class Tests_CGM1:
        def test_lowerlimb(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM1\\kad-med-Options\\"

            # calibration
            filenameNoExt = "static"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM1_calibration.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")

            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")


            # fitting
            filenameNoExt = "Gait1"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM1_fitting.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")


            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")


    class Tests_CGM11:
        def test_lowerlimb(self):
            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM1.1\\pyCGM2 lower limb CGM1 medial\\"

            # calibration
            filenameNoExt = "pyCGM2 lower limb CGM1 medial Static"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM11_calibration.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")

            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")


            # fitting
            filenameNoExt = "pyCGM2 lower limb CGM1 medial Walking01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM11_fitting.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")


            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")


    class Tests_CGM21:
        def test_lowerlimb(self):

            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.1\\pyCGM2 lower limb CGM21\\"

            # calibration
            filenameNoExt = "pyCGM2 lower limb CGM21 Static"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM21_calibration.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")

            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")


            # fitting
            filenameNoExt = "pyCGM2 lower limb CGM21 Walking01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM21_fitting.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")


            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")

    class Tests_CGM22:
        def test_lowerlimb(self):

            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.2\\pyCGM2 lower limb CGM22\\"

            # calibration
            filenameNoExt = "pyCGM2 lower limb CGM22 Static"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM22_calibration.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")

            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")


            # fitting
            filenameNoExt = "pyCGM2 lower limb CGM22 Walking01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM22_fitting.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")


            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")

    class Tests_CGM23:
        def test_lowerlimb(self):

            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.3\\pyCGM2 lower limb CGM23\\"

            # calibration
            filenameNoExt = "pyCGM2 lower limb CGM23 Static"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM23_calibration.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")

            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")


            # fitting
            filenameNoExt = "pyCGM2 lower limb CGM23 Walking01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM23_fitting.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")


            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")

    class Tests_CGM24:
        def test_lowerlimb(self):

            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.4\\pyCGM2 lower limb CGM23\\"

            # calibration
            filenameNoExt = "pyCGM2 lower limb CGM24 Static"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM24_calibration.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")

            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")


            # fitting
            filenameNoExt = "pyCGM2 lower limb CGM24 Walking01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM24_fitting.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")


            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")


    class Tests_CGM25:
        def test_lowerlimb(self):

            NEXUS = ViconNexus.ViconNexus()


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.5\\pyCGM2_FullBody_CGM25\\"

            # calibration
            filenameNoExt = "CGM2_Static_01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM25_calibration.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")

            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")


            # fitting
            filenameNoExt = "CGM2_Walk_02"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("nexus_CGM25_fitting.exe")
            NEXUS.SaveTrial(30)

            acq = btkTools.smartReader(DATA_PATH + filenameNoExt + ".c3d")
            acqRef = btkTools.smartReader(DATA_PATH + filenameNoExt + ".2.c3d")

            testingUtils.test_point_compareToRef(acqRef,acq,"LPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"LFootProgressAngles")


            testingUtils.test_point_compareToRef(acqRef,acq,"RPelvisAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RHipAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RKneeAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RAnkleAngles")
            testingUtils.test_point_compareToRef(acqRef,acq,"RFootProgressAngles")
