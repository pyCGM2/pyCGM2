# coding: utf-8
# pytest -s --log-cli-level=INFO --disable-pytest-warnings  test_NexusCgm.py::Test_CGM1::test_fullBody
import pyCGM2
import pyCGM2; LOGGER = pyCGM2.LOGGER

import pytest
import numpy as np
import os
import matplotlib.pyplot as plt

from viconnexusapi import ViconNexus

try:
    from viconnexusapi import ViconNexus
    NEXUS = ViconNexus.ViconNexus()
except:
    LOGGER.logger.warning("No Nexus connection")
else :


    class Test_CGM1:
        def test_fullBody(self):


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM1\\pyCGM2_FullBody_CGM1_KADmed\\"
            filenameNoExt = "Static"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM1.0 Calibration")
            NEXUS.SaveTrial(30)

            filenameNoExt = "Capture 01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM1.0 Fitting")
            NEXUS.SaveTrial(30)

    class Test_CGM11:
        def test_fullBody(self):


            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM1.1\\pyCGM2_FullBody_CGM1_medial\\"
            filenameNoExt = "FullBody CGM2 data Cal 01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM1.1 Calibration")
            NEXUS.SaveTrial(30)

            filenameNoExt = "Capture 01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM1.1 Fitting")
            NEXUS.SaveTrial(30)

    class Test_CGM21:
        def test_fullBody(self):

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.1\\pyCGM2_FullBody_CGM21\\"
            filenameNoExt = "FullBody CGM2 data Cal 01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM2.1 Calibration")
            NEXUS.SaveTrial(30)

            filenameNoExt = "Capture 01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM2.1 Fitting")
            NEXUS.SaveTrial(30)

    class Test_CGM22:
        def test_fullBody(self):

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.2\\pyCGM2_FullBody_CGM22\\"
            filenameNoExt = "FullBody CGM2 data Cal 01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM2.2 Calibration")
            NEXUS.SaveTrial(30)

            filenameNoExt = "Capture 01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM2.2 Fitting")
            NEXUS.SaveTrial(30)

    class Test_CGM23:
        def test_fullBody(self):

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.3\\pyCGM2_FullBody_CGM23\\"
            filenameNoExt = "FullBody CGM2 data Cal 01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM2.3 Calibration")
            NEXUS.SaveTrial(30)

            filenameNoExt = "Capture 01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM2.3 Fitting")
            NEXUS.SaveTrial(30)




    class Test_CGM24:
        def test_fullBody(self):

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.4\\pyCGM2_FullBody_CGM24\\"
            filenameNoExt = "FullBody CGM2 data Cal 01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM2.4 Calibration")
            NEXUS.SaveTrial(30)

            filenameNoExt = "Capture 01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM2.4 Fitting")
            NEXUS.SaveTrial(30)

    class Test_CGM25:
        def test_fullBody(self):

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.5\\pyCGM2_FullBody_CGM24\\"
            filenameNoExt = "CGM2_Static_01"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM2.5 Calibration")
            NEXUS.SaveTrial(30)

            filenameNoExt = "CGM2_Walk_02"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS CGM2.5 Fitting")
            NEXUS.SaveTrial(30)

    class Test_plots:

        def test_stp(self):

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM23-Patient\\Session 1\\"

            filenameNoExt = "20240214-EC-PONC-F-NNNN04"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS Plots STP")
            NEXUS.SaveTrial(30)


        def test_kinematics(self):

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM23-Patient\\Session 1\\"

            filenameNoExt = "20240214-EC-PONC-F-NNNN04"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS Plots Kinematics Normalized")
            os.system("pyCGM2.exe NEXUS Plots Kinematics Temporal")
            os.system("pyCGM2.exe NEXUS Plots Kinematics MAP")
            NEXUS.SaveTrial(30)

        def test_kinetics(self):

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM23-Patient\\Session 1\\"

            filenameNoExt = "20240214-EC-PONC-F-NNNN04"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS Plots Kinetics Normalized")
            #os.system("pyCGM2.exe NEXUS Plots Kinetics Temporal")
            NEXUS.SaveTrial(30)

        def test_reaction(self):

            DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM23-Patient\\Session 1\\"

            filenameNoExt = "20240214-EC-PONC-F-NNNN04"
            NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
            os.system("pyCGM2.exe NEXUS Plots Reaction Temporal")
            NEXUS.SaveTrial(30)