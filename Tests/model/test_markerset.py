# -*- coding: utf-8 -*-
from pyCGM2.Model.CGM2 import cgm,cgm2
import numpy as np

class tests_getTrackingMarkers():

    @classmethod
    def CGM1(cls):
        TRACKING_MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

        model=cgm.CGM1LowerLimbs()
        model.configure()
        trackingMarkers = model.getTrackingMarkers()

        np.testing.assert_equal(trackingMarkers,
                                TRACKING_MARKERS)
    @classmethod
    def CGM11(cls):
        TRACKING_MARKERS = ["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

        model=cgm.CGM1LowerLimbs()
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




if __name__ == "__main__":


    # tests_getTrackingMarkers.CGM1()
    # tests_getTrackingMarkers.CGM11()
    # tests_getTrackingMarkers.CGM21()
    # tests_getTrackingMarkers.CGM22()
    tests_getTrackingMarkers.CGM23()
    tests_getTrackingMarkers.CGM24()
