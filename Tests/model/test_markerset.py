# -*- coding: utf-8 -*-
import pyCGM2
from pyCGM2.Tools import btkTools

from pyCGM2.Model.CGM2 import cgm,cgm2
import numpy as np


class tests_getTrackingMarkers():

    @classmethod
    def CGM1(cls):
        TRACKING_MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

        model=cgm.CGM1
        model.configure()
        trackingMarkers = model.getTrackingMarkers()

        np.testing.assert_equal(trackingMarkers,
                                TRACKING_MARKERS)
    @classmethod
    def CGM11(cls):
        TRACKING_MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

        model=cgm.CGM1
        model.configure()
        trackingMarkers = model.getTrackingMarkers()

        np.testing.assert_equal(trackingMarkers,
                                TRACKING_MARKERS)
    @classmethod
    def CGM21(cls):
        TRACKING_MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

        model=cgm2.CGM2_1LowerLimbs()
        model.configure()
        trackingMarkers = model.getTrackingMarkers()

        np.testing.assert_equal(trackingMarkers,
                                TRACKING_MARKERS)

    @classmethod
    def CGM22(cls):
        TRACKING_MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

        model=cgm2.CGM2_2LowerLimbs()
        model.configure()
        trackingMarkers = model.getTrackingMarkers()

        np.testing.assert_equal(trackingMarkers,
                                TRACKING_MARKERS)

    @classmethod
    def CGM23(cls):
        TRACKING_MARKERS = ["LASI", "RASI","RPSI", "LPSI",
                   "LTHI","LKNE","LTHAP","LTHAD",
                   "LTIB","LANK","LTIAP","LTIAD",
                   "LHEE","LTOE",
                   "RTHI","RKNE","RTHAP","RTHAD",
                   "RTIB","RANK","RTIAP","RTIAD",
                   "RHEE","RTOE"]

        model=cgm2.CGM2_3LowerLimbs()
        model.configure()
        trackingMarkers = model.getTrackingMarkers()

        np.testing.assert_equal(trackingMarkers,
                                TRACKING_MARKERS)

    @classmethod
    def CGM24(cls):
        TRACKING_MARKERS = ["LASI", "RASI","RPSI", "LPSI",
                   "LTHI","LKNE","LTHAP","LTHAD",
                   "LTIB","LANK","LTIAP","LTIAD",
                   "LHEE","LTOE","LFMH","LVMH",
                   "RTHI","RKNE","RTHAP","RTHAD",
                   "RTIB","RANK","RTIAP","RTIAD",
                   "RHEE","RTOE","RFMH","RVMH"]

        model=cgm2.CGM2_4LowerLimbs()
        model.configure()
        trackingMarkers = model.getTrackingMarkers()

        np.testing.assert_equal(trackingMarkers,
                                TRACKING_MARKERS)



class tests_getStaticMarkers():

    @classmethod
    def CGM1(cls):
        TRACKING_MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]
        STATIC_MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]


        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "native.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)


        model=cgm.CGM1
        model.configure()
        trackingMarkers = model.getTrackingMarkers()
        staticMarkers = model.getStaticMarkers(dcm)

        np.testing.assert_equal(staticMarkers,
                                STATIC_MARKERS)


    @classmethod
    def CGM1_KADs(cls):
        TRACKING_MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]
        STATIC_MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LTIB","LANK","LHEE","LTOE","RTHI","RTIB","RANK","RHEE","RTOE"] + \
                        cgm.CGM.KAD_MARKERS["Left"] + cgm.CGM.KAD_MARKERS["Right"]


        DATA_PATH = pyCGM2.TEST_DATA_PATH + "operations\\staticMarkerConfig\\CGM\\"
        staticFilename = "KAD-both.c3d"

        acq = btkTools.smartReader(str(DATA_PATH+staticFilename))
        dcm = cgm.CGM.detectCalibrationMethods(acq)


        model=cgm.CGM1
        model.configure()
        trackingMarkers = model.getTrackingMarkers()
        staticMarkers = model.getStaticMarkers(dcm)

        np.testing.assert_equal(staticMarkers,STATIC_MARKERS)

    # TODO cases (KAD-med) -(kneeMed)- Knee and ankleMed




if __name__ == "__main__":


    tests_getTrackingMarkers.CGM1()
    tests_getTrackingMarkers.CGM11()
    tests_getTrackingMarkers.CGM21()
    tests_getTrackingMarkers.CGM22()
    tests_getTrackingMarkers.CGM23()
    tests_getTrackingMarkers.CGM24()


    tests_getStaticMarkers.CGM1()
    tests_getStaticMarkers.CGM1_KADs()
