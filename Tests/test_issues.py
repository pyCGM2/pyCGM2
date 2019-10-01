# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_issues.py::TestIssueModel

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
from pyCGM2.Utils import testingUtils
import pytest

class TestIssueModel:
    def test_issue_signAbdAddOffset(self):

        DATA_PATH = MAIN_PATH = pyCGM2.TEST_DATA_PATH + "Datatests from Vicon\\issues\\Nick-issueAbdAddOffset\\pyCGM2_FullBody_PiG_KADmed - myTest\\"
        staticFilename = "Static.c3d"

        acqStatic = btkTools.smartReader(str(DATA_PATH +  staticFilename))

        markerDiameter=14
        leftFlatFoot = True
        rightFlatFoot = True
        headStraight = True
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(str(DATA_PATH + "Nick.vsk"))
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
