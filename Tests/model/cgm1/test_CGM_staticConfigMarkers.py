# -*- coding: utf-8 -*-
import numpy as np

import pyCGM2

from pyCGM2 import enums
from pyCGM2.Tools import btkTools
from pyCGM2.Model.CGM2 import cgm


class tests():

    @classmethod
    def nativeCGM(cls):

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "native.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        np.testing.assert_equal(smc["left"],enums.CgmStaticMarkerConfig.Native )
        np.testing.assert_equal(smc["right"],enums.CgmStaticMarkerConfig.Native )

    @classmethod
    def KADs(cls):

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KAD-both.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        np.testing.assert_equal(smc["left"],enums.CgmStaticMarkerConfig.KAD )
        np.testing.assert_equal(smc["right"],enums.CgmStaticMarkerConfig.KAD )

    @classmethod
    def KAD_left(cls):

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KAD-onlyLeft.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        np.testing.assert_equal(smc["left"],enums.CgmStaticMarkerConfig.KAD )
        np.testing.assert_equal(smc["right"],enums.CgmStaticMarkerConfig.Native )

    @classmethod
    def KAD_right(cls):

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KAD-onlyRight.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        np.testing.assert_equal(smc["left"],enums.CgmStaticMarkerConfig.Native )
        np.testing.assert_equal(smc["right"],enums.CgmStaticMarkerConfig.KAD )


    @classmethod
    def KADmeds(cls):

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KADmed-both.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        np.testing.assert_equal(smc["left"],enums.CgmStaticMarkerConfig.KADmed )
        np.testing.assert_equal(smc["right"],enums.CgmStaticMarkerConfig.KADmed )

    @classmethod
    def KADmed_left(cls):

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KADmed-onlyLeft.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        np.testing.assert_equal(smc["left"],enums.CgmStaticMarkerConfig.KADmed )
        np.testing.assert_equal(smc["right"],enums.CgmStaticMarkerConfig.Native )

    @classmethod
    def KADmed_right(cls):

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KADmed-onlyRight.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        np.testing.assert_equal(smc["left"],enums.CgmStaticMarkerConfig.Native )
        np.testing.assert_equal(smc["right"],enums.CgmStaticMarkerConfig.KADmed )


    @classmethod
    def Meds(cls):

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "medMed.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        np.testing.assert_equal(smc["left"],enums.CgmStaticMarkerConfig.KneeAnkleMed )
        np.testing.assert_equal(smc["right"],enums.CgmStaticMarkerConfig.KneeAnkleMed )

    @classmethod
    def Meds_left(cls):

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "medMed-onlyLeft.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        np.testing.assert_equal(smc["left"],enums.CgmStaticMarkerConfig.KneeAnkleMed )
        np.testing.assert_equal(smc["right"],enums.CgmStaticMarkerConfig.Native )

    @classmethod
    def Meds_right(cls):

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "medMed-onlyRight.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        np.testing.assert_equal(smc["left"],enums.CgmStaticMarkerConfig.Native )
        np.testing.assert_equal(smc["right"],enums.CgmStaticMarkerConfig.KneeAnkleMed )


    @classmethod
    def KneeMeds(cls):

        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "medKnee.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acq)

        np.testing.assert_equal(smc["left"],enums.CgmStaticMarkerConfig.KneeMed )
        np.testing.assert_equal(smc["right"],enums.CgmStaticMarkerConfig.KneeMed )






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
