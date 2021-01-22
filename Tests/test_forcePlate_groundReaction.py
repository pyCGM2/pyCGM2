# -*- coding: utf-8 -*-
# from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_forcePlate_groundReaction.py::Test_groundReactionForcePlate::test_sample0
import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.ForcePlates import forceplates

import logging
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2.Lib.CGM import  cgm1
from pyCGM2.Model import modelFilters
from pyCGM2.Eclipse import vskTools
from pyCGM2.Utils import testingUtils


class Test_groundReactionForcePlate():

    def test_sample0(self):

        DATA_PATH =  pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\fullBody-native-noOptions\\"
        staticFilename = "static.c3d"

        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vsk = vskTools.Vsk(DATA_PATH + "New Subject.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        # calibration according CGM1
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

        # no fitting operation, only checking of forceplateAssembly
        gaitFilename="gait1.c3d"
        acqGait = btkTools.smartReader(DATA_PATH +  gaitFilename)
        mfpa = None


        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGait,mappedForcePlate)
        logging.warning("Manual Force plate assignment : %s" %mappedForcePlate)


        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGait,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix=pointSuffix)

        # #rmsValue =numeric.rms( acqGait.GetPoint("LGroundReactionForce").GetValues()[init:end,:] -   acqGait.GetPoint("LGroundReactionForce_test").GetValues()[init:end,:], axis = 0)

        # testingUtils.plotComparisonOfPoint(acqGait,"LGroundReactionForce","test",init=420, end = 470)
        testingUtils.test_point_rms(acqGait,"LGroundReactionForce","LGroundReactionForce_test",0.5,init=420, end = 470)
        # testingUtils.plotComparisonOfPoint(acqGait,"RGroundReactionForce","test",init=464, end = 520)
        testingUtils.test_point_rms(acqGait,"RGroundReactionForce","RGroundReactionForce_test",0.5,init=464, end = 520)
