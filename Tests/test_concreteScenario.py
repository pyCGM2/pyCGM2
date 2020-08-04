# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_concreteScenario.py::TestData_withNoFP::test_CGM1_FullBody_noOptions_noFP
# pytest -s --disable-pytest-warnings  test_concreteScenario.py::Test_DifferentStaticDynamicMarkerSet::test_CGM1_FullBody_noOptions_uncorrectPelvisMarker
import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Utils import files

import logging
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2 import enums
from pyCGM2.Lib.CGM import  cgm1,cgm2_4
from pyCGM2.Model import modelFilters
from pyCGM2.Eclipse import vskTools
from pyCGM2.Utils import testingUtils,utils


class Test_Data_withNoFP:
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

        # btkTools.smartWriter(acqGait, "gait1-pyCGM2modelled.c3d")


class Test_DifferentStaticDynamicMarkerSet:
    def test_CGM1_FullBody_noOptions_uncorrectUpperLimbMarker(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Scenarii\different static and dynamic marker set\CGM1-fullBody\\"
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
        logging.info("Static Calibration -----> Done")

        # case 1 -  one marker on the upper limb misses
        gaitFilename="gait1_noRFIN.c3d"
        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        assert model.m_bodypart  == enums.BodyPart.LowerLimbTrunk


    def test_CGM1_FullBody_noOptions_uncorrectThoraxMarker(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Scenarii\\different static and dynamic marker set\CGM1-fullBody\\"
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
        logging.info("Static Calibration -----> Done")

        gaitFilename="gait1_noSTRN.c3d"
        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        assert model.m_bodypart  == enums.BodyPart.LowerLimb

    def test_CGM1_FullBody_noOptions_uncorrectPelvisMarker(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Scenarii\\different static and dynamic marker set\CGM1-fullBody\\"
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
        logging.info("Static Calibration -----> Done")

        gaitFilename="gait1_noLASI.c3d"
        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            None,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        assert model.m_bodypart  == enums.BodyPart.UpperLimb


    def test_CGM24_FullBody_noOptions_uncorrectUpperLimbMarker(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Scenarii\\different static and dynamic marker set\\CGM24-fullBody\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)


        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "PN07.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_4-pyCGM2.settings")
        hjcMethod = settings["Calibration"]["HJC"]

        translators = files.getTranslators(DATA_PATH,"CGM2_4.translators")
        # if not translators:  translators = settings["Translators"]


        model,finalAcqStatic = cgm2_4.calibrate(DATA_PATH,
            staticFilename,
            translators,
            settings,
            required_mp,
            optional_mp,
            False,
            leftFlatFoot,
            rightFlatFoot,
            headStraight,
            markerDiameter,
            hjcMethod,
            pointSuffix,
            displayCoordinateSystem=True)

        # btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        logging.info("Static Calibration -----> Done")

        gaitFilename="gait1_noFIN.c3d"

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm2_4.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,
            settings,
            False,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        assert model.m_bodypart  == enums.BodyPart.LowerLimbTrunk

    def test_CGM24_FullBody_noOptions_uncorrectThoraxMarker(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Scenarii\\different static and dynamic marker set\\CGM24-fullBody\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)


        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "PN07.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_4-pyCGM2.settings")
        hjcMethod = settings["Calibration"]["HJC"]

        translators = files.getTranslators(DATA_PATH,"CGM2_4.translators")
        # if not translators:  translators = settings["Translators"]


        model,finalAcqStatic = cgm2_4.calibrate(DATA_PATH,
            staticFilename,
            translators,
            settings,
            required_mp,
            optional_mp,
            False,
            leftFlatFoot,
            rightFlatFoot,
            headStraight,
            markerDiameter,
            hjcMethod,
            pointSuffix,
            displayCoordinateSystem=True)

        # btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        logging.info("Static Calibration -----> Done")

        gaitFilename="gait1_noCLAV.c3d"

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm2_4.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,
            settings,
            False,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        assert model.m_bodypart  == enums.BodyPart.LowerLimb

    def test_CGM24_FullBody_noOptions_uncorrectLowerLimbMarker(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Scenarii\\different static and dynamic marker set\\CGM24-fullBody\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)


        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vskFile = vskTools.getVskFiles(DATA_PATH)
        vsk = vskTools.Vsk(DATA_PATH + "PN07.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_4-pyCGM2.settings")
        hjcMethod = settings["Calibration"]["HJC"]

        translators = files.getTranslators(DATA_PATH,"CGM2_4.translators")
        # if not translators:  translators = settings["Translators"]


        model,finalAcqStatic = cgm2_4.calibrate(DATA_PATH,
            staticFilename,
            translators,
            settings,
            required_mp,
            optional_mp,
            False,
            leftFlatFoot,
            rightFlatFoot,
            headStraight,
            markerDiameter,
            hjcMethod,
            pointSuffix,
            displayCoordinateSystem=True)

        # btkTools.smartWriter(finalAcqStatic, str( staticFilename[:-4]+"-pyCGM2modelled.c3d"))
        logging.info("Static Calibration -----> Done")

        gaitFilename="gait1_noLASI.c3d"

        mfpa = None
        reconstructFilenameLabelled = gaitFilename

        acqGait = cgm2_4.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,
            settings,
            False,
            markerDiameter,
            pointSuffix,
            mfpa,
            enums.MomentProjection.Proximal,
            displayCoordinateSystem=True)

        assert model.m_bodypart  == enums.BodyPart.UpperLimb
