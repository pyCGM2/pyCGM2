# coding: utf-8
#pytest -s --mpl --disable-pytest-warnings  test_mp.py::Test_CGM::test_cgm1

import numpy as np
from docutils import SettingsSpec
import pytest
from pyCGM2.Report import normativeDatasets

import matplotlib.pyplot as plt

import pyCGM2; LOGGER = pyCGM2.LOGGER

import pyCGM2

from pyCGM2 import enums
from pyCGM2.Lib.CGM import  cgm1
from pyCGM2.Nexus import vskTools

from pyCGM2.Utils import files

import ipdb

class Test_customMp:


    def test_mp(self):

        path = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\Settings\\"
        mpSettings = files.openFile(path, "mp.settings")
        ipdb.set_trace()

class Test_CGM:

    def test_cgm1(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\\CGM1\\fullBody-native-noOptions-customMP\\"
        staticFilename = "static.c3d"

        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vsk = vskTools.Vsk(DATA_PATH + "New Subject.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        model,finalAcqStatic,error = cgm1.calibrate(DATA_PATH,
            staticFilename,
            None,
            required_mp,
            optional_mp,
            leftFlatFoot,
            rightFlatFoot,
            headStraight,
            markerDiameter,
            pointSuffix,
            displayCoordinateSystem=True)


        # btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        # LOGGER.logger.info("Static Calibration -----> Done")

        gaitFilename="gait1.Distal.c3d"

        mfpa = "RLXX"
        reconstructFilenameLabelled = gaitFilename

        acqGait,error = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Distal,
            displayCoordinateSystem=True)

        np.testing.assert_equal(model.getSegment("Pelvis").m_bsp["inertia"], np.array([[1,2,3],[4,5,6],[7,8,9]]))
        np.testing.assert_equal(model.getSegment("Pelvis").m_bsp["com"], np.array([0,0,3]))
