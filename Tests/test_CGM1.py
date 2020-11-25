# -*- coding: utf-8 -*-
# from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_CGM1.py::Test_FullBody::test_FullBody_noOptions

import os
import logging

import numpy as np

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2 import enums
from pyCGM2.Lib.CGM import  cgm1
from pyCGM2.Tools import btkTools
from pyCGM2.Eclipse import vskTools
from pyCGM2.Utils import testingUtils


class Test_FullBody:
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


        testingUtils.test_offset(model.mp_computed["LeftThighRotationOffset"],acqStatic,"LThighRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightThighRotationOffset"],acqStatic,"RThighRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftShankRotationOffset"],acqStatic,"LShankRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightShankRotationOffset"],acqStatic,"RShankRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftTibialTorsionOffset"],acqStatic,"LTibialTorsion", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightTibialTorsionOffset"],acqStatic,"RTibialTorsion", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftAnkleAbAddOffset"],acqStatic,"LAnkleAbAdd", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightAnkleAbAddOffset"],acqStatic,"RAnkleAbAdd", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftStaticPlantFlexOffset"],acqStatic,"LStaticPlantFlex", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightStaticPlantFlexOffset"],acqStatic,"RStaticPlantFlex", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftStaticRotOffset"],acqStatic,"LStaticRotOff", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightStaticRotOffset"],acqStatic,"RStaticRotOff", decimal=3)


        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LHJC").m_desc ,"Davis")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RHJC").m_desc ,"Davis")
        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"Chord")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"Chord")
        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"Chord")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"Chord")
        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"Chord")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"Chord")
        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"Chord")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"Chord")


        testingUtils.test_point(finalAcqStatic,"LPelvisAngles","LPelvisAngles_test",decimal = 3)
        testingUtils.test_point(finalAcqStatic,"RPelvisAngles","RPelvisAngles_test",decimal = 3)
        testingUtils.test_point(finalAcqStatic,"LHipAngles","LHipAngles_test",decimal = 3)
        testingUtils.test_point(finalAcqStatic,"LKneeAngles","LKneeAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"LAnkleAngles","LAnkleAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"RHipAngles","RHipAngles_test",decimal = 3)
        testingUtils.test_point(finalAcqStatic,"RKneeAngles","RKneeAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"RAnkleAngles","RAnkleAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"LFootProgressAngles","LFootProgressAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"RFootProgressAngles","RFootProgressAngles_test",decimal = 2)

        testingUtils.test_point(finalAcqStatic,"LThoraxAngles","LThoraxAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"RThoraxAngles","RThoraxAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"LSpineAngles","LSpineAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"RSpineAngles","RSpineAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"LShoulderAngles","LShoulderAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"RShoulderAngles","RShoulderAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"LElbowAngles","LElbowAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"RElbowAngles","RElbowAngles_test",decimal = 2)
        testingUtils.test_point(finalAcqStatic,"LHeadAngles","LHeadAngles_test",decimal = 3)
        testingUtils.test_point(finalAcqStatic,"RHeadAngles","RHeadAngles_test",decimal = 3)


        # btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        # logging.info("Static Calibration -----> Done")

        gaitFilename="gait1.c3d"

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        testingUtils.test_point(acqGait,"LPelvisAngles","LPelvisAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"RPelvisAngles","RPelvisAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"LHipAngles","LHipAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"LKneeAngles","LKneeAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"LAnkleAngles","LAnkleAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RHipAngles","RHipAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"RKneeAngles","RKneeAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RAnkleAngles","RAnkleAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"LFootProgressAngles","LFootProgressAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RFootProgressAngles","RFootProgressAngles_test",decimal = 2)

        testingUtils.test_point(acqGait,"LThoraxAngles","LThoraxAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RThoraxAngles","RThoraxAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"LSpineAngles","LSpineAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RSpineAngles","RSpineAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"LShoulderAngles","LShoulderAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RShoulderAngles","RShoulderAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"LElbowAngles","LElbowAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RElbowAngles","RElbowAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"LHeadAngles","LHeadAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"RHeadAngles","RHeadAngles_test",decimal = 3)
        # testingUtils.test_point(acqGait,"LWristAngles","LWristAngles_test",decimal = 3) fail on z!
        # testingUtils.test_point(acqGait,"RWristAngles","RWristAngles_test",decimal = 3) fail on Z!
        # testingUtils.test_point(acqGait,"CentreOfMass","CentreOfMass_test",decimal = 3)

        btkTools.smartAppendPoint(acqGait,"headCOM_py",model.getSegment("Head").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"ThoraxCOM_py",model.getSegment("Thorax").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LhumCOM_py",model.getSegment("Left UpperArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LforeCom_py",model.getSegment("Left ForeArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"LhandCom_py",model.getSegment("Left Hand").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RhumCOM_py",model.getSegment("Right UpperArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RforeCom_py",model.getSegment("Right ForeArm").getComTrajectory())
        btkTools.smartAppendPoint(acqGait,"RhandCom_py",model.getSegment("Right Hand").getComTrajectory())


    # testingUtils.plotValuesComparison(acqGait.GetPoint("LeftHumerusCOM").GetValues(), model.getSegment("Left UpperArm").getComTrajectory())
    # testingUtils.plotComparisonofPoint(finalAcqStatic,"RAnkleAngles","test")

class Test_LowerBody():

    def test_KadMed_options(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\kad-med-Options\\"
        staticFilename = "static.c3d"
        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)

        markerDiameter=14
        leftFlatFoot = True
        rightFlatFoot = True
        headStraight = True
        pointSuffix = "test"

        # vskFile = vskTools.getVskFiles(DATA_PATH)
        # vsk = vskTools.Vsk(str(DATA_PATH + "PIG-KAD.vsk"))
        # required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        required_mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 40,
        'LeftElbowWidth' : 74,
        'LeftWristWidth' : 55 ,
        'LeftHandThickness' : 34 ,
        'RightShoulderOffset'   : 40,
        'RightElbowWidth' : 74,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 34}

        optional_mp={
        'InterAsisDistance'   :  0,#0,
        'LeftAsisTrocanterDistance' :  0,#0,
        'LeftTibialTorsion' :  0,#0,
        'LeftThighRotation' :  0,#0,
        'LeftShankRotation' : 0,#0,ipdb
        'RightAsisTrocanterDistance' : 0,#0,
        'RightTibialTorsion' :  0,#0,
        'RightThighRotation' :  0,#0,
        'RightShankRotation' : 0}

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

        testingUtils.test_offset(model.mp_computed["LeftThighRotationOffset"],acqStatic,"LThighRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightThighRotationOffset"],acqStatic,"RThighRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftShankRotationOffset"],acqStatic,"LShankRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightShankRotationOffset"],acqStatic,"RShankRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftTibialTorsionOffset"],acqStatic,"LTibialTorsion", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightTibialTorsionOffset"],acqStatic,"RTibialTorsion", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftAnkleAbAddOffset"],acqStatic,"LAnkleAbAdd", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightAnkleAbAddOffset"],acqStatic,"RAnkleAbAdd", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftStaticPlantFlexOffset"],acqStatic,"LStaticPlantFlex", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightStaticPlantFlexOffset"],acqStatic,"RStaticPlantFlex", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftStaticRotOffset"],acqStatic,"LStaticRotOff", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightStaticRotOffset"],acqStatic,"RStaticRotOff", decimal=3)

        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LHJC").m_desc ,"Davis")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RHJC").m_desc ,"Davis")
        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"mid")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"mid")
        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"mid")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"mid")

        # btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        # logging.info("Static Calibration -----> Done")

        gaitFilename="gait1.c3d"

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        testingUtils.test_point(acqGait,"LHipAngles","LHipAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"LKneeAngles","LKneeAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"LAnkleAngles","LAnkleAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RHipAngles","RHipAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"RKneeAngles","RKneeAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RAnkleAngles","RAnkleAngles_test",decimal = 2)



        # testingUtils.plotComparisonofPoint(finalAcqStatic,"RAnkleAngles","test")

    def test_KadMed_noOptions(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\kad-med-noOptions\\"
        staticFilename = "static.c3d"
        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)

        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        # vskFile = vskTools.getVskFiles(DATA_PATH)
        # vsk = vskTools.Vsk(str(DATA_PATH + "PIG-KAD.vsk"))
        # required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        required_mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 40,
        'LeftElbowWidth' : 74,
        'LeftWristWidth' : 55 ,
        'LeftHandThickness' : 34 ,
        'RightShoulderOffset'   : 40,
        'RightElbowWidth' : 74,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 34}

        optional_mp={
        'InterAsisDistance'   :  0,#0,
        'LeftAsisTrocanterDistance' :  0,#0,
        'LeftTibialTorsion' :  0,#0,
        'LeftThighRotation' :  0,#0,
        'LeftShankRotation' : 0,#0,ipdb
        'RightAsisTrocanterDistance' : 0,#0,
        'RightTibialTorsion' :  0,#0,
        'RightThighRotation' :  0,#0,
        'RightShankRotation' : 0}

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

        testingUtils.test_offset(model.mp_computed["LeftThighRotationOffset"],acqStatic,"LThighRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightThighRotationOffset"],acqStatic,"RThighRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftShankRotationOffset"],acqStatic,"LShankRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightShankRotationOffset"],acqStatic,"RShankRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftTibialTorsionOffset"],acqStatic,"LTibialTorsion", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightTibialTorsionOffset"],acqStatic,"RTibialTorsion", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftAnkleAbAddOffset"],acqStatic,"LAnkleAbAdd", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightAnkleAbAddOffset"],acqStatic,"RAnkleAbAdd", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftStaticPlantFlexOffset"],acqStatic,"LStaticPlantFlex", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightStaticPlantFlexOffset"],acqStatic,"RStaticPlantFlex", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftStaticRotOffset"],acqStatic,"LStaticRotOff", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightStaticRotOffset"],acqStatic,"RStaticRotOff", decimal=3)

        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LHJC").m_desc ,"Davis")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RHJC").m_desc ,"Davis")
        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"mid")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"mid")
        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"mid")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"mid")

        # btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        # logging.info("Static Calibration -----> Done")

        gaitFilename="gait1.c3d"

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        testingUtils.test_point(acqGait,"LHipAngles","LHipAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"LKneeAngles","LKneeAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"LAnkleAngles","LAnkleAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RHipAngles","RHipAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"RKneeAngles","RKneeAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RAnkleAngles","RAnkleAngles_test",decimal = 2)



    def test_Kad_options(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\kad-options\\"
        staticFilename = "static.c3d"
        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)

        markerDiameter=14
        leftFlatFoot = True
        rightFlatFoot = True
        headStraight = True
        pointSuffix = "test"

        # vskFile = vskTools.getVskFiles(DATA_PATH)
        # vsk = vskTools.Vsk(str(DATA_PATH + "PIG-KAD.vsk"))
        # required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        required_mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 40,
        'LeftElbowWidth' : 74,
        'LeftWristWidth' : 55 ,
        'LeftHandThickness' : 34 ,
        'RightShoulderOffset'   : 40,
        'RightElbowWidth' : 74,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 34}

        optional_mp={
        'InterAsisDistance'   :  0,#0,
        'LeftAsisTrocanterDistance' :  0,#0,
        'LeftTibialTorsion' :  0,#0,
        'LeftThighRotation' :  0,#0,
        'LeftShankRotation' : 0,#0,
        'RightAsisTrocanterDistance' : 0,#0,
        'RightTibialTorsion' :  0,#0,
        'RightThighRotation' :  0,#0,
        'RightShankRotation' : 0}

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


        testingUtils.test_offset(model.mp_computed["LeftThighRotationOffset"],acqStatic,"LThighRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightThighRotationOffset"],acqStatic,"RThighRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftShankRotationOffset"],acqStatic,"LShankRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightShankRotationOffset"],acqStatic,"RShankRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftTibialTorsionOffset"],acqStatic,"LTibialTorsion", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightTibialTorsionOffset"],acqStatic,"RTibialTorsion", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftAnkleAbAddOffset"],acqStatic,"LAnkleAbAdd", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightAnkleAbAddOffset"],acqStatic,"RAnkleAbAdd", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftStaticPlantFlexOffset"],acqStatic,"LStaticPlantFlex", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightStaticPlantFlexOffset"],acqStatic,"RStaticPlantFlex", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftStaticRotOffset"],acqStatic,"LStaticRotOff", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightStaticRotOffset"],acqStatic,"RStaticRotOff", decimal=3)

        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LHJC").m_desc ,"Davis")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RHJC").m_desc ,"Davis")
        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"KAD")

        # btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        # logging.info("Static Calibration -----> Done")

        gaitFilename="gait1.c3d"

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        testingUtils.test_point(acqGait,"LHipAngles","LHipAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"LKneeAngles","LKneeAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"LAnkleAngles","LAnkleAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RHipAngles","RHipAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"RKneeAngles","RKneeAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RAnkleAngles","RAnkleAngles_test",decimal = 2)

    def test_Kad_noOptions(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\kad-noOptions\\"
        staticFilename = "static.c3d"
        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)

        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        # vskFile = vskTools.getVskFiles(DATA_PATH)
        # vsk = vskTools.Vsk(str(DATA_PATH + "PIG-KAD.vsk"))
        # required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        required_mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 40,
        'LeftElbowWidth' : 74,
        'LeftWristWidth' : 55 ,
        'LeftHandThickness' : 34 ,
        'RightShoulderOffset'   : 40,
        'RightElbowWidth' : 74,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 34}

        optional_mp={
        'InterAsisDistance'   :  0,#0,
        'LeftAsisTrocanterDistance' :  0,#0,
        'LeftTibialTorsion' :  0,#0,
        'LeftThighRotation' :  0,#0,
        'LeftShankRotation' : 0,#0,
        'RightAsisTrocanterDistance' : 0,#0,
        'RightTibialTorsion' :  0,#0,
        'RightThighRotation' :  0,#0,
        'RightShankRotation' : 0}

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


        testingUtils.test_offset(model.mp_computed["LeftThighRotationOffset"],acqStatic,"LThighRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightThighRotationOffset"],acqStatic,"RThighRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftShankRotationOffset"],acqStatic,"LShankRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightShankRotationOffset"],acqStatic,"RShankRotation", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftTibialTorsionOffset"],acqStatic,"LTibialTorsion", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightTibialTorsionOffset"],acqStatic,"RTibialTorsion", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftAnkleAbAddOffset"],acqStatic,"LAnkleAbAdd", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightAnkleAbAddOffset"],acqStatic,"RAnkleAbAdd", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftStaticPlantFlexOffset"],acqStatic,"LStaticPlantFlex", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightStaticPlantFlexOffset"],acqStatic,"RStaticPlantFlex", decimal=3)
        testingUtils.test_offset(model.mp_computed["LeftStaticRotOffset"],acqStatic,"LStaticRotOff", decimal=3)
        testingUtils.test_offset(model.mp_computed["RightStaticRotOffset"],acqStatic,"RStaticRotOff", decimal=3)

        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LHJC").m_desc ,"Davis")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RHJC").m_desc ,"Davis")
        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"KAD")

        # btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        # logging.info("Static Calibration -----> Done")

        gaitFilename="gait1.c3d"

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        testingUtils.test_point(acqGait,"LHipAngles","LHipAngles_test",decimal = 3)
        testingUtils.test_point(acqGait,"LKneeAngles","LKneeAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"LAnkleAngles","LAnkleAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RHipAngles","RHipAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RKneeAngles","RKneeAngles_test",decimal = 2)
        testingUtils.test_point(acqGait,"RAnkleAngles","RAnkleAngles_test",decimal = 2)
