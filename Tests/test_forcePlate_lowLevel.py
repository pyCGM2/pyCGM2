# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_forcePlate_lowLevel.py::Test_correctForcePlate::test_correctForcePlateType5
# pytest -s --disable-pytest-warnings  test_forcePlate_lowLevel.py::Test_groundReactionForcePlate::test_sample0
"""
Created on Thu Jul 07 15:14:18 2016

@author: aaa34169
"""
import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Utils import files

import logging
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2 import enums
from pyCGM2.Lib.CGM import  cgm1
from pyCGM2.Model import modelFilters
from pyCGM2.Eclipse import vskTools
from pyCGM2.Utils import testingUtils,utils


class Test_correctForcePlate():

    def test_correctForcePlateType5(self):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ForcePlate\\ForcePlateTypeManagement\\"
        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"LowLevel\\ForcePlate\\ForcePlateTypeManagement\\"
        files.createDir(DATA_PATH_OUT)


        btkAcq = btkTools.smartReader(MAIN_PATH + "HUG_gait_type5.c3d")

        forceplates.correctForcePlateType5(btkAcq)

        btkTools.smartWriter(btkAcq,DATA_PATH_OUT+ "HUG_gait_type5.c3d")

class Test_groundReactionForcePlate():

    def test_sample0(self):

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

        testingUtils.plotComparisonOfPoint(acqGait,"RGroundReactionForce","test")
        testingUtils.plotComparisonOfPoint(acqGait,"RGroundReactionMoment","test")
