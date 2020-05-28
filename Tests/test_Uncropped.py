# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# pytest -s --disable-pytest-warnings  test_Gap.py::TestFullBody::test_FullBody_noOptions

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
from pyCGM2.Utils import files
import pytest

class TestFullBody:
    def test_FullBody_noOptions(self):
        DATA_PATH = MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\uncropped data\\cgm1\\"
        staticFilename = "static.c3d"

        DATA_PATH_OUT = pyCGM2.TEST_DATA_PATH_OUT+"LowLevel\\uncropped data\\cgm1\\"
        files.createDir(DATA_PATH_OUT)

        markerDiameter=14
        leftFlatFoot = True
        rightFlatFoot = True
        headStraight = True
        pointSuffix = ""

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

        acqGait0 = btkTools.smartReader(DATA_PATH +  gaitFilename)
        trackingMarkers = model.getTrackingMarkers(acqGait0)
        validFrames,vff,vlf = btkTools.findValidFrames(acqGait0,trackingMarkers)


        mfpa = None
        reconstructFilenameLabelled = gaitFilename


        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        btkTools.smartWriter(acqGait, DATA_PATH_OUT+"//gait1-processed.c3d")
