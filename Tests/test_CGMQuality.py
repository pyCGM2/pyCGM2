# -*- coding: utf-8 -*-
# from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_CGMQuality.py::Test_CGM1_wandPlanarAngles::test_FullBody_noOptions

import os
import pyCGM2; LOGGER = pyCGM2.LOGGER

import numpy as np

import pyCGM2

from pyCGM2 import enums
from pyCGM2.Lib.CGM import  cgm1
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import vskTools
from pyCGM2.Utils import testingUtils


class Test_CGM1_wandPlanarAngles:
    def test_FullBody_noOptions(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\fullBody-native-Options\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)

        markerDiameter=14
        leftFlatFoot = True
        rightFlatFoot = True
        headStraight = True
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
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

        btkTools.smartWriter(finalAcqStatic,"calibtest.c3d")        

        gaitFilename="gait1.c3d"

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait,error = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        btkTools.smartWriter(acqGait,"fittingTest.c3d")


    def test_LowerBody_pelvisGap(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "gapScenario\\pyCGM2 lower limb CGM23-Gaps\\"

        staticFilename = "pyCGM2 lower limb CGM23 Static.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)

        markerDiameter=14
        leftFlatFoot = True
        rightFlatFoot = True
        headStraight = True
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "Nick.vsk")
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

        btkTools.smartWriter(finalAcqStatic,"calibtest.c3d")        

        gaitFilename="pyCGM2 lower limb CGM23 Walking01-PelvisGap.c3d"

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait,error = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        btkTools.smartWriter(acqGait,"fittingTest.c3d")