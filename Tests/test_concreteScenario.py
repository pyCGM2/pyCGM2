# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_concreteScenario.py::TestData_withNoFP::test_CGM1_FullBody_noOptions_noFP
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


class TestData_withNoFP:
    def test_CGM1_FullBody_noOptions_noFP(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\fullBody-native-noOptions - noFP\\"
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

        btkTools.smartWriter(acqGait, "gait1-pyCGM2modelled.c3d")
