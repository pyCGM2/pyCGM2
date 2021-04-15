# coding: utf-8
# pytest -s --log-cli-level=INFO --disable-pytest-warnings  test_NexusCgm.py::Test_CGM1::test_fullBody
import pyCGM2

import pytest
import numpy as np
import os

from viconnexusapi import ViconNexus

NEXUS = ViconNexus.ViconNexus()

class Test_CGM1:
    def test_fullBody(self):


        DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM1\\pyCGM2_FullBody_CGM1_KADmed\\"
        filenameNoExt = "Static"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM1_Calibration.exe")
        NEXUS.SaveTrial(30)

        filenameNoExt = "Capture 01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM1_Fitting.exe")
        NEXUS.SaveTrial(30)

class Test_CGM11:
    def test_fullBody(self):


        DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM1.1\\pyCGM2_FullBody_CGM1_medial\\"
        filenameNoExt = "FullBody CGM2 data Cal 01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM11_Calibration.exe")
        NEXUS.SaveTrial(30)

        filenameNoExt = "Capture 01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM11_Fitting.exe")
        NEXUS.SaveTrial(30)

class Test_CGM21:
    def test_fullBody(self):

        DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.1\\pyCGM2_FullBody_CGM21\\"
        filenameNoExt = "FullBody CGM2 data Cal 01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM21_Calibration.exe")
        NEXUS.SaveTrial(30)

        filenameNoExt = "Capture 01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM21_Fitting.exe")
        NEXUS.SaveTrial(30)

class Test_CGM22:
    def test_fullBody(self):

        DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.2\\pyCGM2_FullBody_CGM22\\"
        filenameNoExt = "FullBody CGM2 data Cal 01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM22_Calibration.exe")
        NEXUS.SaveTrial(30)

        filenameNoExt = "Capture 01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM22_Fitting.exe")
        NEXUS.SaveTrial(30)

class Test_CGM23:
    def test_fullBody(self):

        DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.3\\pyCGM2_FullBody_CGM23\\"
        filenameNoExt = "FullBody CGM2 data Cal 01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM23_Calibration.exe")
        NEXUS.SaveTrial(30)

        filenameNoExt = "Capture 01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM23_Fitting.exe")
        NEXUS.SaveTrial(30)

class Test_CGM24:
    def test_fullBody(self):

        DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.4\\pyCGM2_FullBody_CGM24\\"
        filenameNoExt = "FullBody CGM2 data Cal 01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM24_Calibration.exe")
        NEXUS.SaveTrial(30)

        filenameNoExt = "Capture 01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM24_Fitting.exe")
        NEXUS.SaveTrial(30)

class Test_CGM25:
    def test_fullBody(self):

        DATA_PATH =  pyCGM2.TEST_DATA_PATH+"Nexus\\CGM2.5\\pyCGM2_FullBody_CGM24\\"
        filenameNoExt = "CGM2_Static_01"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM25_Calibration.exe")
        NEXUS.SaveTrial(30)

        filenameNoExt = "CGM2_Walk_02"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        os.system("Nexus_CGM25_Fitting.exe")
        NEXUS.SaveTrial(30)
