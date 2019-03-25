# -*- coding: utf-8 -*-
import numpy as np
import pdb
import logging
import json

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.WARNING)


# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model import modelFilters, modelDecorator, frame
import pyCGM2.enums as pyCGM2Enums
from collections import OrderedDict
from pyCGM2.Utils import files




class translatorScenario_tests():

    @classmethod
    def scenario1Test(cls):
        """
        basic test.

        the pyCGM2 marker is not in the c3d. it points to you own label which is not a pyCGM2 marker
        ( e.g : LASI points to LeftASI)

        return:
           => own markers and pyGM2 markers are both IN the final c3d

        """


        contents24 ="""
            Translators:
                LASI: LeftASI
                RASI: RightASI
                LPSI: LeftPSI
                RPSI: RightPSI
                RTHI: None
                RKNE: None
                RTIAP: None
                RTIAD: None
                RTIB: None
                RANK: None
                RTIAP: None
                RTIAD: None
                RHEE: None
                RSMH: None
                RTOE: None
                RFMH: None
                RVMH: None
                LTHI: None
                LKNE: None
                LTHAP: None
                LTHAD: None
                LTIB: None
                LANK: None
                LTIAP: None
                LTIAD: None
                LHEE: None
                LSMH: None
                LTOE: None
                LFMH: None
                LVMH: None
                RKNM: None
                LKNM: None
                RMED: None
                LMED: None
                C7: None
                T10: None
                CLAV: None
                STRN: None
                LFHD: None
                LBHD: None
                RFHD: None
                RBHD: None
                LSHO: None
                LELB: None
                LWRB: None
                LWRA: None
                LFIN: None
                RSHO: None
                RELB: None
                RWRB: None
                RWRA: None
                RFIN: None
            """
        translators = files.openYaml(None,None,contents24)

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\Translators\\scenario1\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        acqStatic2 =  btkTools.applyTranslators(acqStatic,translators["Translators"])

        btkTools.smartWriter(acqStatic2,"scenario1Test.c3d")

        np.testing.assert_equal(acqStatic2.GetPoint("LeftASI").GetValues(), acqStatic2.GetPoint("LASI").GetValues())
        np.testing.assert_equal(acqStatic2.GetPoint("RightASI").GetValues(), acqStatic2.GetPoint("RASI").GetValues())
        np.testing.assert_equal(acqStatic2.GetPoint("LeftPSI").GetValues(), acqStatic2.GetPoint("LPSI").GetValues())
        np.testing.assert_equal(acqStatic2.GetPoint("RightPSI").GetValues(), acqStatic2.GetPoint("RPSI").GetValues())


    @classmethod
    def scenario2Test(cls):
        """

        the pyCGM2 translator is IN the c3d. it points to your own label which is NOT a pyCGM2 translator list
        ( e.g : LTHI points to LTHLD)

        return:
            the translator and your own markers point to similar values (e.g LTHI amd LTHD point same values)
            a new marker suffixed with _origin for keeping a trace of the translator found in the c3d  ( eg LTHi renamed LTHI_origin )

        """


        contents24 ="""
            Translators:
                LASI: None
                RASI: None
                LPSI: None
                RPSI: None
                RTHI: None
                RKNE: None
                RTIAP: None
                RTIAD: None
                RTIB: None
                RANK: None
                RTIAP: None
                RTIAD: None
                RHEE: None
                RSMH: None
                RTOE: None
                RFMH: None
                RVMH: None
                LTHI: LTHLD
                LKNE: None
                LTHAP: None
                LTHAD: None
                LTIB: None
                LANK: None
                LTIAP: None
                LTIAD: None
                LHEE: None
                LSMH: None
                LTOE: None
                LFMH: None
                LVMH: None
                RKNM: None
                LKNM: None
                RMED: None
                LMED: None
                C7: None
                T10: None
                CLAV: None
                STRN: None
                LFHD: None
                LBHD: None
                RFHD: None
                RBHD: None
                LSHO: None
                LELB: None
                LWRB: None
                LWRA: None
                LFIN: None
                RSHO: None
                RELB: None
                RWRB: None
                RWRA: None
                RFIN: None
            """
        translators = files.openYaml(None,None,contents24)

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\Translators\\scenario2\\"
        staticFilename = "staticAlana.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))


        acqStatic2 =  btkTools.applyTranslators(acqStatic,translators["Translators"])


        btkTools.smartWriter(acqStatic2,"scenario2Test.c3d")

        np.testing.assert_equal(acqStatic2.GetPoint("LTHI").GetValues(), acqStatic2.GetPoint("LTHLD").GetValues())
        np.testing.assert_equal(acqStatic2.GetPoint("LTHI_origin").GetValues(), acqStatic.GetPoint("LTHI").GetValues())


    @classmethod
    def scenario3Test(cls):
        """

        the pyCGM2 marker is IN the c3d. it points to another pyCGM2 marker which is IN c3d
        ( e.g : LTHI points to LTHAP)

        return:
            the translator and your own markers point to similar values (e.g LTHI amd LTAP point same values)
            a new marker suffixed with _origin for keeping a trace of the translator found in the c3d  ( eg LTHi renamed LTHI_origin )


        """


        contents24 ="""
            Translators:
                LASI: None
                RASI: None
                LPSI: None
                RPSI: None
                RTHI: None
                RKNE: None
                RTIAP: None
                RTIAD: None
                RTIB: None
                RANK: None
                RTIAP: None
                RTIAD: None
                RHEE: None
                RSMH: None
                RTOE: None
                RFMH: None
                RVMH: None
                LTHI: LTHAD
                LKNE: None
                LTHAP: None
                LTHAD: None
                LTIB: None
                LANK: None
                LTIAP: None
                LTIAD: None
                LHEE: None
                LSMH: None
                LTOE: None
                LFMH: None
                LVMH: None
                RKNM: None
                LKNM: None
                RMED: None
                LMED: None
                C7: None
                T10: None
                CLAV: None
                STRN: None
                LFHD: None
                LBHD: None
                RFHD: None
                RBHD: None
                LSHO: None
                LELB: None
                LWRB: None
                LWRA: None
                LFIN: None
                RSHO: None
                RELB: None
                RWRB: None
                RWRA: None
                RFIN: None
            """
        translators = files.openYaml(None,None,contents24)

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\Translators\\scenario3\\"
        staticFilename = "staticAlana.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))


        acqStatic2 =  btkTools.applyTranslators(acqStatic,translators["Translators"])


        btkTools.smartWriter(acqStatic2,"scenario3Test.c3d")

        np.testing.assert_equal(acqStatic2.GetPoint("LTHI").GetValues(), acqStatic2.GetPoint("LTHAD").GetValues())
        np.testing.assert_equal(acqStatic2.GetPoint("LTHI_origin").GetValues(), acqStatic.GetPoint("LTHI").GetValues())

    @classmethod
    def scenario4Test(cls):
        """

        you want to swp to markers
        ( e.g : LTHI swap with LTHAD)

        return:
            in the final c3d, both markers are swapped. there are no _origin markers
        """


        contents24 ="""
            Translators:
                LASI: None
                RASI: None
                LPSI: None
                RPSI: None
                RTHI: None
                RKNE: None
                RTIAP: None
                RTIAD: None
                RTIB: None
                RANK: None
                RTIAP: None
                RTIAD: None
                RHEE: None
                RSMH: None
                RTOE: None
                RFMH: None
                RVMH: None
                LTHI: LTHAD
                LKNE: None
                LTHAP: None
                LTHAD: LTHI
                LTIB: None
                LANK: None
                LTIAP: None
                LTIAD: None
                LHEE: None
                LSMH: None
                LTOE: None
                LFMH: None
                LVMH: None
                RKNM: None
                LKNM: None
                RMED: None
                LMED: None
                C7: None
                T10: None
                CLAV: None
                STRN: None
                LFHD: None
                LBHD: None
                RFHD: None
                RBHD: None
                LSHO: None
                LELB: None
                LWRB: None
                LWRA: None
                LFIN: None
                RSHO: None
                RELB: None
                RWRB: None
                RWRA: None
                RFIN: None
            """
        translators = files.openYaml(None,None,contents24)

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\Translators\\scenario4\\"
        staticFilename = "staticAlana.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))


        acqStatic2 =  btkTools.applyTranslators(acqStatic,translators["Translators"])


        btkTools.smartWriter(acqStatic2,"scenario4Test.c3d")

        np.testing.assert_equal(acqStatic2.GetPoint("LTHI").GetValues(), acqStatic.GetPoint("LTHAD").GetValues())
        np.testing.assert_equal(acqStatic2.GetPoint("LTHAD").GetValues(), acqStatic.GetPoint("LTHI").GetValues())


class concreteScenario_tests():

    @classmethod
    def cgm1_sacrum(cls):
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\Translators\\cgm1-sacr\\"
        staticFilename = "static.c3d"

        translators = files.getTranslators(MAIN_PATH, translatorType = "CGM1.translators")

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        acqStatic2 =  btkTools.applyTranslators(acqStatic,translators)


        btkTools.smartWriter(acqStatic2,"staticCGM1Sacrum.c3d")

        np.testing.assert_equal(acqStatic2.GetPoint("LPSI").GetValues(), acqStatic.GetPoint("SACR").GetValues())
        np.testing.assert_equal(acqStatic2.GetPoint("RPSI").GetValues(), acqStatic.GetPoint("SACR").GetValues())


    @classmethod
    def translators24_initiateWith_25markerset(cls):


        contents24 ="""
            Translators:
                LASI: None
                RASI: None
                LPSI: None
                RPSI: None
                RTHI: None
                RKNE: None
                RTIAP: None
                RTIAD: None
                RTIB: None
                RANK: None
                RTIAP: None
                RTIAD: None
                RHEE: None
                RSMH: None
                RTOE: None
                RFMH: None
                RVMH: None
                LTHI: None
                LKNE: None
                LTHAP: None
                LTHAD: None
                LTIB: None
                LANK: None
                LTIAP: None
                LTIAD: None
                LHEE: None
                LSMH: None
                LTOE: None
                LFMH: None
                LVMH: None
                RKNM: None
                LKNM: None
                RMED: None
                LMED: None
                C7: T2
                T10: None
                CLAV: None
                STRN: CLAV
                LFHD: GLAB
                LBHD: LMAS
                RFHD: GLAB
                RBHD: RMAS
                LSHO: None
                LELB: None
                LWRB: None
                LWRA: None
                LFIN: None
                RSHO: None
                RELB: None
                RWRB: None
                RWRA: None
                RFIN: None
            """
        translators = files.openYaml(None,None,contents24)

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\Translators\\cgm2.5\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))


        acqStatic2 =  btkTools.applyTranslators(acqStatic,translators["Translators"])


        btkTools.smartWriter(acqStatic2,"translators24_initiateWith_25markerset.c3d")


        np.testing.assert_equal(acqStatic2.GetPoint("C7").GetValues(), acqStatic.GetPoint("T2").GetValues())
        np.testing.assert_equal(acqStatic2.GetPoint("LFHD").GetValues(), acqStatic.GetPoint("GLAB").GetValues())
        np.testing.assert_equal(acqStatic2.GetPoint("RFHD").GetValues(), acqStatic.GetPoint("GLAB").GetValues())
        np.testing.assert_equal(acqStatic2.GetPoint("LBHD").GetValues(), acqStatic.GetPoint("LMAS").GetValues())
        np.testing.assert_equal(acqStatic2.GetPoint("RBHD").GetValues(), acqStatic.GetPoint("RMAS").GetValues())

if __name__ == "__main__":



    translatorScenario_tests.scenario1Test()
    translatorScenario_tests.scenario2Test()
    translatorScenario_tests.scenario3Test()
    translatorScenario_tests.scenario4Test()


    concreteScenario_tests.cgm1_sacrum()
    concreteScenario_tests.translators24_initiateWith_25markerset()
