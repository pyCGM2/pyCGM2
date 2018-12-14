# -*- coding: utf-8 -*-
import numpy as np

import pyCGM2

from pyCGM2 import enums
from pyCGM2.Tools import btkTools
from pyCGM2.Model.CGM2 import cgm


class tests():

    @classmethod
    def nativeCGM(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "native.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)

        np.testing.assert_equal(dcm["Left Knee"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Right Knee"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Left Ankle"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Right Ankle"],enums.JointCalibrationMethod.Basic )

    @classmethod
    def KADs(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KAD-both.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)

        np.testing.assert_equal(dcm["Left Knee"],enums.JointCalibrationMethod.KAD )
        np.testing.assert_equal(dcm["Right Knee"],enums.JointCalibrationMethod.KAD )
        np.testing.assert_equal(dcm["Left Ankle"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Right Ankle"],enums.JointCalibrationMethod.Basic )

    @classmethod
    def KAD_left(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KAD-onlyLeft.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)

        np.testing.assert_equal(dcm["Left Knee"],enums.JointCalibrationMethod.KAD )
        np.testing.assert_equal(dcm["Right Knee"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Left Ankle"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Right Ankle"],enums.JointCalibrationMethod.Basic )

    @classmethod
    def KAD_right(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KAD-onlyRight.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)

        np.testing.assert_equal(dcm["Left Knee"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Right Knee"],enums.JointCalibrationMethod.KAD )
        np.testing.assert_equal(dcm["Left Ankle"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Right Ankle"],enums.JointCalibrationMethod.Basic )


    @classmethod
    def KADmeds(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KADmed-both.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)

        np.testing.assert_equal(dcm["Left Knee"],enums.JointCalibrationMethod.KAD )
        np.testing.assert_equal(dcm["Right Knee"],enums.JointCalibrationMethod.KAD )
        np.testing.assert_equal(dcm["Left Ankle"],enums.JointCalibrationMethod.Medial )
        np.testing.assert_equal(dcm["Right Ankle"],enums.JointCalibrationMethod.Medial )

    @classmethod
    def KADmed_left(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KADmed-onlyLeft.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)

        np.testing.assert_equal(dcm["Left Knee"],enums.JointCalibrationMethod.KAD )
        np.testing.assert_equal(dcm["Right Knee"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Left Ankle"],enums.JointCalibrationMethod.Medial )
        np.testing.assert_equal(dcm["Right Ankle"],enums.JointCalibrationMethod.Basic )

    @classmethod
    def KADmed_right(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KADmed-onlyRight.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)

        np.testing.assert_equal(dcm["Left Knee"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Right Knee"],enums.JointCalibrationMethod.KAD )
        np.testing.assert_equal(dcm["Left Ankle"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Right Ankle"],enums.JointCalibrationMethod.Medial )


    @classmethod
    def Meds(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "medMed.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        dcm = cgm.CGM.detectCalibrationMethods(acq)

        np.testing.assert_equal(dcm["Left Knee"],enums.JointCalibrationMethod.Medial )
        np.testing.assert_equal(dcm["Right Knee"],enums.JointCalibrationMethod.Medial )
        np.testing.assert_equal(dcm["Left Ankle"],enums.JointCalibrationMethod.Medial )
        np.testing.assert_equal(dcm["Right Ankle"],enums.JointCalibrationMethod.Medial )

    @classmethod
    def Meds_left(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "medMed-onlyLeft.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)

        np.testing.assert_equal(dcm["Left Knee"],enums.JointCalibrationMethod.Medial )
        np.testing.assert_equal(dcm["Right Knee"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Left Ankle"],enums.JointCalibrationMethod.Medial )
        np.testing.assert_equal(dcm["Right Ankle"],enums.JointCalibrationMethod.Basic )

    @classmethod
    def Meds_right(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "medMed-onlyRight.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)

        np.testing.assert_equal(dcm["Left Knee"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Right Knee"],enums.JointCalibrationMethod.Medial )
        np.testing.assert_equal(dcm["Left Ankle"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Right Ankle"],enums.JointCalibrationMethod.Medial )


    @classmethod
    def KneeMeds(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "medKnee.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)

        np.testing.assert_equal(dcm["Left Knee"],enums.JointCalibrationMethod.Medial )
        np.testing.assert_equal(dcm["Right Knee"],enums.JointCalibrationMethod.Medial )
        np.testing.assert_equal(dcm["Left Ankle"],enums.JointCalibrationMethod.Basic )
        np.testing.assert_equal(dcm["Right Ankle"],enums.JointCalibrationMethod.Basic )




if __name__ == "__main__":

    tests.nativeCGM()
    tests.KADs()
    tests.KAD_left()
    tests.KAD_right()
    tests.KADmeds()
    tests.KADmed_left()
    tests.KADmed_right()
    tests.Meds()
    tests.Meds_left()
    tests.Meds_right()
    tests.KneeMeds()
