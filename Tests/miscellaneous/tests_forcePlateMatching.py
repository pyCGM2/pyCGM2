# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 15:14:18 2016

@author: aaa34169
"""

import logging
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.ForcePlates import forceplates

import numpy as np



class test_matchedFootPlatForm():

    @classmethod
    def twoPF(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1
        gaitFilename="walking_oppositeX_2pf.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)

        print mappedForcePlate
        if mappedForcePlate!="LX":
            raise Exception ("uncorrected force plate matching")

        # --- Motion 2
        gaitFilename="walking_X_2pf.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)

        print mappedForcePlate
        if mappedForcePlate!="XX":
            raise Exception ("uncorrected force plate matching")

    @classmethod
    def threePF(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1
        gaitFilename="walking_Y_3pf.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)


        if mappedForcePlate!="RLR":
            raise Exception ("uncorrected force plate matching")

    @classmethod
    def threePF_patho(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1
        gaitFilename="walking_pathoY_onlyRight.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)


        if mappedForcePlate!="RRR":
            raise Exception ("uncorrected force plate matching")


class test_matchedFootPlatForm_difficultCases():
    @classmethod
    def fourPF_PF3misfunction(cls):
        """
        FP#3 misfunction. Zeroing was not performed. Thus an offset superior to the threshold occurs from the beggining
        """

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1
        gaitFilename="walking-X-4pf.c3d"


        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)

        print mappedForcePlate
        if mappedForcePlate!="LRXX":
            raise Exception ("uncorrected force plate matching")

    @classmethod
    def sixForcePlate_overlayFP4and5(cls):
        """
        Basic algorithm detects right foot with force #5.
        The main problem is overlay of left foot on FP#4 and #5 influences joint kinetics of the cycle beginning around frame 170 on force plate 3.
        This articfact corruptes joint moment during swing phase
        """


        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\forceplates\\footAssignement\\"

        # --- Motion 1
        gaitFilename="gait trial 01.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)

        if mappedForcePlate!="RLRLXX":
            raise Exception ("uncorrected force plate matching")

class test_manualAssigment():
    @classmethod
    def threePF_wrongAssigmenent(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1
        gaitFilename="walking_Y_3pf.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)

        try:
            assignedMappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait,mfpa="XX")
        except Exception, errormsg:
            np.testing.assert_string_equal(errormsg.args[0],"[pyCGM2] number of assigned force plate inferior to the number of force plate number. Your assignment should have  3 letters at least")


    @classmethod
    def threePF_assigmenentCases(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1
        gaitFilename="walking_Y_3pf.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        correctAssigmenent = "RLR"

        assignedMappedForcePlate1 = forceplates.matchingFootSideOnForceplate(acqGait,mfpa="XXX")
        np.testing.assert_string_equal(assignedMappedForcePlate1,"XXX")

        assignedMappedForcePlate2 = forceplates.matchingFootSideOnForceplate(acqGait,mfpa="XXA")
        np.testing.assert_string_equal(assignedMappedForcePlate2,"XXR")

        assignedMappedForcePlate3 = forceplates.matchingFootSideOnForceplate(acqGait,mfpa="AAA")
        np.testing.assert_string_equal(assignedMappedForcePlate3,"RLR")

        assignedMappedForcePlate4 = forceplates.matchingFootSideOnForceplate(acqGait,mfpa="AXA")
        np.testing.assert_string_equal(assignedMappedForcePlate4,"RXR")

    @classmethod
    def threePF_mfpaSupNumberForcePlates(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\forceplates\\detectFoot\\"

        # --- Motion 1
        gaitFilename="walking_Y_3pf.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        correctAssigmenent = "RLR"

        assignedMappedForcePlate1 = forceplates.matchingFootSideOnForceplate(acqGait,mfpa="XXXXXX")
        np.testing.assert_string_equal(assignedMappedForcePlate1,"XXX")


        assignedMappedForcePlate2 = forceplates.matchingFootSideOnForceplate(acqGait,mfpa="XXALR")
        np.testing.assert_string_equal(assignedMappedForcePlate2,"XXR")

        assignedMappedForcePlate3 = forceplates.matchingFootSideOnForceplate(acqGait,mfpa="AAALR")
        np.testing.assert_string_equal(assignedMappedForcePlate3,"RLR")

        assignedMappedForcePlate4 = forceplates.matchingFootSideOnForceplate(acqGait,mfpa="AXAXXX")
        np.testing.assert_string_equal(assignedMappedForcePlate4,"RXR")


if __name__ == "__main__":
    plt.close("all")

    logging.info("######## BASIC CASE ######")
    test_matchedFootPlatForm.twoPF()
    test_matchedFootPlatForm.threePF()
    test_matchedFootPlatForm.threePF_patho()

    logging.info("######## DIFFICULT CASES ######")
    test_matchedFootPlatForm_difficultCases.fourPF_PF3misfunction()
    test_matchedFootPlatForm_difficultCases.sixForcePlate_overlayFP4and5()

    logging.info("######## MANUAL ASSIGMENT ######")
    test_manualAssigment.threePF_wrongAssigmenent()
    test_manualAssigment.threePF_assigmenentCases()
    test_manualAssigment.threePF_mfpaSupNumberForcePlates()
