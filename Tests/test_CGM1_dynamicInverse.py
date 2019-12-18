# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# pytest -s --disable-pytest-warnings  test_CGM1_dynamicInverse.py::TestFullBody::test_FullBody_noOptions_distal

import ipdb
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2 import enums
from pyCGM2.Lib.CGM import  cgm1
from pyCGM2.Tools import btkTools
from pyCGM2.Eclipse import vskTools
from pyCGM2.Utils import testingUtils,utils
import pytest

class TestFullBody:


    def test_FullBody_noOptions_proximal(self):
        DATA_PATH = MAIN_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\fullBody-native-noOptions\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)


        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "New Subject.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        model,finalAcqStatic = cgm1.calibrate(DATA_PATH,
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
        # logging.info("Static Calibration -----> Done")

        gaitFilename="gait1.Proximal.c3d"

        mfpa = "RLXX"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        testingUtils.plotComparisonOfPoint(acqGait,"LAnkleForce","test")
        testingUtils.plotComparisonOfPoint(acqGait,"RAnkleForce","test")

        gaitFilename="gait2.Proximal.c3d"

        mfpa = "LRLX"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)
            
        testingUtils.plotComparisonOfPoint(acqGait,"LAnkleForce","test")
        testingUtils.plotComparisonOfPoint(acqGait,"RAnkleForce","test")

    def test_FullBody_noOptions_distal(self):
        DATA_PATH = MAIN_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\fullBody-native-noOptions\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)


        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "New Subject.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        model,finalAcqStatic = cgm1.calibrate(DATA_PATH,
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
        # logging.info("Static Calibration -----> Done")

        gaitFilename="gait1.Distal.c3d"

        mfpa = "RLXX"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Distal,
            displayCoordinateSystem=True)

        testingUtils.plotComparisonOfPoint(acqGait,"LAnkleForce","test")
        testingUtils.plotComparisonOfPoint(acqGait,"RAnkleForce","test")

        gaitFilename="gait2.Distal.c3d"

        mfpa = "LRLX"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Distal,
            displayCoordinateSystem=True)

        testingUtils.plotComparisonOfPoint(acqGait,"LAnkleForce","test",title="Global_LAnkF")
        testingUtils.plotComparisonOfPoint(acqGait,"RAnkleForce","test",title="Global_LAnkF")


    def test_FullBody_noOptions_global(self):
        DATA_PATH = MAIN_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\fullBody-native-noOptions\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)


        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "New Subject.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        model,finalAcqStatic = cgm1.calibrate(DATA_PATH,
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
        # logging.info("Static Calibration -----> Done")

        gaitFilename="gait1.Global.c3d"

        mfpa = "RLXX"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Global,
            displayCoordinateSystem=True)

        testingUtils.plotComparisonOfPoint(acqGait,"LAnkleForce","test",title="Global_LAnkF")
        testingUtils.plotComparisonOfPoint(acqGait,"RAnkleForce","test",title="Global_LAnkF")


        gaitFilename="gait2.Global.c3d"

        mfpa = "LRLX"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Global,
            displayCoordinateSystem=True)

        testingUtils.plotComparisonOfPoint(acqGait,"LAnkleForce","test",title="Global_LAnkF")
        testingUtils.plotComparisonOfPoint(acqGait,"RAnkleForce","test",title="Global_LAnkF")
