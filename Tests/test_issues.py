# coding: utf-8
from __future__ import unicode_literals

# pytest -s --disable-pytest-warnings  test_issues.py::TestStephenM::test_issue_signAbdAddOffset


import ipdb
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2 import enums
from pyCGM2.Lib.CGM import  cgm1,cgm2_4
from pyCGM2.Tools import btkTools
from pyCGM2.Eclipse import vskTools
from pyCGM2.Utils import testingUtils,files,utils
import pytest
from pyCGM2 import btk
from pyCGM2.Processing import progressionFrame

class TestStephenM:
    def test_issue_signAbdAddOffset(self):


        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Issues\\StephenM\\signedAbdAdd-KadMed\\"
        staticFilename = "Static.c3d"
        acqStatic = btkTools.smartReader(str(DATA_PATH +  staticFilename))

        markerDiameter=14
        leftFlatFoot = True
        rightFlatFoot = True
        headStraight = True
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "Nick.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)
        # required_mp={
        # 'Bodymass'   : 65.0,
        # 'LeftLegLength' : 800.0,
        # 'RightLegLength' : 800.0 ,
        # 'LeftKneeWidth' : 120.0,
        # 'RightKneeWidth' : 120.0,
        # 'LeftAnkleWidth' : 100.0,
        # 'RightAnkleWidth' : 100.0,
        # 'LeftSoleDelta' : 0,
        # 'RightSoleDelta' : 0,
        # 'LeftShoulderOffset'   : 40,
        # 'LeftElbowWidth' : 74,
        # 'LeftWristWidth' : 55 ,
        # 'LeftHandThickness' : 34 ,
        # 'RightShoulderOffset'   : 40,
        # 'RightElbowWidth' : 74,
        # 'RightWristWidth' : 55 ,
        # 'RightHandThickness' : 34}
        #
        # optional_mp={
        # 'InterAsisDistance'   :  0,#0,
        # 'LeftAsisTrocanterDistance' :  0,#0,
        # 'LeftTibialTorsion' :  0,#0,
        # 'LeftThighRotation' :  0,#0,
        # 'LeftShankRotation' : 0,#0,ipdb
        # 'RightAsisTrocanterDistance' : 0,#0,
        # 'RightTibialTorsion' :  0,#0,
        # 'RightThighRotation' :  0,#0,
        # 'RightShankRotation' : 0}


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

        testingUtils.test_offset(model.mp_computed["LeftThighRotationOffset"],acqStatic,"LThighRotation", decimal=1) # not enought accurate but unsignificant
        testingUtils.test_offset(model.mp_computed["RightThighRotationOffset"],acqStatic,"RThighRotation", decimal=1)
        testingUtils.test_offset(model.mp_computed["LeftShankRotationOffset"],acqStatic,"LShankRotation", decimal=1)
        testingUtils.test_offset(model.mp_computed["RightShankRotationOffset"],acqStatic,"RShankRotation", decimal=1)
        testingUtils.test_offset(model.mp_computed["LeftTibialTorsionOffset"],acqStatic,"LTibialTorsion", decimal=1)
        testingUtils.test_offset(model.mp_computed["RightTibialTorsionOffset"],acqStatic,"RTibialTorsion", decimal=1)
        testingUtils.test_offset(model.mp_computed["LeftAnkleAbAddOffset"],acqStatic,"LAnkleAbAdd", decimal=1)
        testingUtils.test_offset(model.mp_computed["RightAnkleAbAddOffset"],acqStatic,"RAnkleAbAdd", decimal=1)
        testingUtils.test_offset(model.mp_computed["LeftStaticPlantFlexOffset"],acqStatic,"LStaticPlantFlex", decimal=1)
        testingUtils.test_offset(model.mp_computed["RightStaticPlantFlexOffset"],acqStatic,"RStaticPlantFlex", decimal=1)
        testingUtils.test_offset(model.mp_computed["LeftStaticRotOffset"],acqStatic,"LStaticRotOff", decimal=1)
        testingUtils.test_offset(model.mp_computed["RightStaticRotOffset"],acqStatic,"RStaticRotOff", decimal=1)


    def test_issue_jointForce_CGM24(self):
        """
        synopsis : inverted sign of the x-component of the joint force with CGM1.1 to CGM2.4
        """

        DATA_PATH = MAIN_PATH = pyCGM2.TEST_DATA_PATH + "Issues\\StephenM\\sign_jointForce_CGM24\\"
        staticFilename = "FullBody CGM2 data Cal 01.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)


        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "Nick.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        settings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_4-pyCGM2.settings")

        model,finalAcqStatic = cgm2_4.calibrate(DATA_PATH,
            staticFilename,
            None,
            settings,
            required_mp,
            optional_mp,
            True,
            leftFlatFoot,
            rightFlatFoot,
            headStraight,
            markerDiameter,
            {"Left": "Hara","Right": "Hara"},
            pointSuffix,
            displayCoordinateSystem=True)


        # btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        # logging.info("Static Calibration -----> Done")

        gaitFilename="Capture 02.Distal.c3d"

        mfpa = "R"
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm2_4.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            settings,
            True,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Distal,
            displayCoordinateSystem=True)

        testingUtils.plotComparisonOfPoint(acqGait,"RAnkleForce","test",title="RAnkleForce PiG - CGM24")
        testingUtils.plotComparisonOfPoint(acqGait,"RKneeForce","test",title="RKneeForce PiG - CGM24")
        testingUtils.plotComparisonOfPoint(acqGait,"RHipForce","test",title="RHipForce PiG - CGM24")






class Test_BrianH:

    def test_progressionFrameIssue(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "\\Issues\\BrianH\\progressionFrame Issue\\"


        gaitFilename="walk09.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

        valSACR=acq.GetPoint(utils.str("SACR")).GetValues()

        btkTools.smartAppendPoint(acq,"RPSI",valSACR,desc="")
        btkTools.smartAppendPoint(acq,"LPSI",valSACR,desc="")


        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)


        gaitFilename="walk11.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        valSACR=acq.GetPoint(utils.str("SACR")).GetValues()

        btkTools.smartAppendPoint(acq,"RPSI",valSACR,desc="")
        btkTools.smartAppendPoint(acq,"LPSI",valSACR,desc="")


        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
