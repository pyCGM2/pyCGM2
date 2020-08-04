# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# pytest -s --disable-pytest-warnings  test_CGM1_dynamicInverse.py::TestFullBody::test_FullBody_noOptions_distal
# pytest -s --disable-pytest-warnings  test_CGM1_dynamicInverse.py::TestLowerBody_progressionY::test_LowerBody_noOptions_proximal
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

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 415, end = 509)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 415, end = 509)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init=415,end=509)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 463, end = 557)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 463, end = 557)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init=463, end = 557)



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

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 273, end = 368)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 273, end = 368)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init = 273, end = 368)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 322, end = 413)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 322, end = 413)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init = 322, end = 413)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right")

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

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 415, end = 509)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 415, end = 509)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init=415,end=509)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 463, end = 557)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 463, end = 557)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init=463, end = 557)

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

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 273, end = 368)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 273, end = 368)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init = 273, end = 368)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 322, end = 413)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 322, end = 413)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init = 322, end = 413)


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

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 415, end = 509)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 415, end = 509)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init=415,end=509)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init=415,end=509)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 463, end = 557)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 463, end = 557)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init=463, end = 557)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init=463, end = 557)


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

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 273, end = 368)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 273, end = 368)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init = 273, end = 368)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 322, end = 413)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 322, end = 413)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init = 322, end = 413)

class TestLowerBody_progressionY:

    def test_LowerBody_noOptions_proximal(self):
        DATA_PATH = MAIN_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\\CGM1\\LowerLimb-medMed_Yprogression\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)

        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "Subject.vsk")
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

        mfpa = "RLX"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 415, end = 509)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 415, end = 509)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init=554, end = 663)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 463, end = 557)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 463, end = 557)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init=480, end = 600)


        gaitFilename="gait2.Proximal.c3d"

        mfpa = "XLR"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 273, end = 368)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 273, end = 368)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init = 273, end = 368)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 322, end = 413)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 322, end = 413)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init = 322, end = 413)

    def test_LowerBody_noOptions_distal(self):
        DATA_PATH = MAIN_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\\CGM1\\LowerLimb-medMed_Yprogression\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)


        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "Subject.vsk")
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

        mfpa = "RLX"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Distal,
            displayCoordinateSystem=True)

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 415, end = 509)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 415, end = 509)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init=554, end = 663)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 463, end = 557)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 463, end = 557)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init=480, end = 600)
        gaitFilename="gait2.Distal.c3d"

        mfpa = "XLR"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Distal,
            displayCoordinateSystem=True)

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 273, end = 368)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 273, end = 368)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init = 273, end = 368)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 322, end = 413)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 322, end = 413)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init = 322, end = 413)


    def test_LowerBody_noOptions_global(self):
        DATA_PATH = MAIN_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\\CGM1\\LowerLimb-medMed_Yprogression\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)


        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "Subject.vsk")
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

        mfpa = "RLX"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Global,
            displayCoordinateSystem=True)

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 415, end = 509)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 415, end = 509)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init=554, end = 663)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init=554, end = 663)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 463, end = 557)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 463, end = 557)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init=480, end = 600)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init=480, end = 600)


        gaitFilename="gait2.Global.c3d"

        mfpa = "XLR"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Global,
            displayCoordinateSystem=True)

        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Left",init = 273, end = 368)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Left",init = 273, end = 368)

        testingUtils.test_point_rms(acqGait,"LAnkleForce","LAnkleForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeForce","LKneeForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipForce","LHipForce_test",0.5,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LAnkleMoment","LAnkleMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LKneeMoment","LKneeMoment_test",60.0,init = 273, end = 368)
        testingUtils.test_point_rms(acqGait,"LHipMoment","LHipMoment_test",60.0,init = 273, end = 368)


        # testingUtils.plotComparison_ForcePanel(acqGait,None,"test","Right",init = 322, end = 413)
        # testingUtils.plotComparison_MomentPanel(acqGait,None,"test","Right",init = 322, end = 413)

        testingUtils.test_point_rms(acqGait,"RAnkleForce","RAnkleForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeForce","RKneeForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipForce","RHipForce_test",0.5,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RAnkleMoment","RAnkleMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RKneeMoment","RKneeMoment_test",60.0,init = 322, end = 413)
        testingUtils.test_point_rms(acqGait,"RHipMoment","RHipMoment_test",60.0,init = 322, end = 413)
