# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:09:22 2016

@author: aaa34169

TODO : these cases are lacking :
 - progression Z lateral axis (X or Y)
 - progression X lateral axis Z
 - progression Y lateral axis Z

"""

import numpy as np
import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)

# pyCGM2
from pyCGM2.Tools import  btkTools,trialTools

from pyCGM2.Processing import progressionFrame

# ---- BTK ------

# Gait
class BtkProgressionTest_gaitTrial():

    @classmethod
    def gaitTrialProgressionX_forward_lateralY(cls):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\progression\\"


        gaitFilename="gait_X_forward.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgression(acq,"LASI")

        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,True)
        np.testing.assert_equal( globalFrame,"XYZ")

        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")


        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrame.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")





    @classmethod
    def gaitTrialProgressionX_backward_lateralY(cls):
        """

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\progression\\"

        gaitFilename="gait_X_backward.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")


        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgression(acq,"LASI")

        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,False)
        np.testing.assert_equal( globalFrame,"XYZ")


        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")


        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrame.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")


    @classmethod
    def gaitTrialProgressionY_forward_lateralX(cls):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\progression\\"


        gaitFilename="gait_Y_forward.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgression(acq,"LASI")

        np.testing.assert_equal( longitudinalAxis,"Y")
        np.testing.assert_equal( forwardProgression,True)
        np.testing.assert_equal( globalFrame,"YXZ")

        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"Y")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"YXZ")

        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrame.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"Y")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"YXZ")



    @classmethod
    def gaitTrialProgressionY_backward_lateralX(cls):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\progression\\"


        gaitFilename="gait_Y_backward.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgression(acq,"LASI")

        np.testing.assert_equal( longitudinalAxis,"Y")
        np.testing.assert_equal( forwardProgression,False)
        np.testing.assert_equal( globalFrame,"YXZ")

        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"Y")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"YXZ")

        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrame.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"Y")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"YXZ")



    @classmethod
    def upperBody_gaitTrialProgressionX_forward_lateralY(cls):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\progression\\"


        gaitFilename="fullBody_GaitX_forward.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgression(acq,"CLAV")

        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,True)
        np.testing.assert_equal( globalFrame,"XYZ")


        pfp = progressionFrame.ThoraxProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")


    @classmethod
    def upperBody_gaitTrialProgressionX_backward_lateralY(cls):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\progression\\"


        gaitFilename="fullBody_GaitX_backward.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgression(acq,"CLAV")

        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,False)
        np.testing.assert_equal( globalFrame,"XYZ")


        pfp = progressionFrame.ThoraxProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")

#--- static
class BtkProgressionTest_static():

    @classmethod
    def gaitTrialProgressionX_forward_lateralY_static(cls):
        """

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\progression\\"

        gaitFilename="static_X.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        valMidAsis=(acq.GetPoint("LASI").GetValues() + acq.GetPoint("RASI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"midASIS",valMidAsis,desc="")


        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acq,["LASI","LPSI","RASI","RPSI"])
        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,True)
        np.testing.assert_equal( globalFrame,"XYZ")


        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")


        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrame.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")




    @classmethod
    def gaitTrialProgressionX_backward_lateralY_static(cls):
        """

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\progression\\"

        gaitFilename="static_X_backward.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        valMidAsis=(acq.GetPoint("LASI").GetValues() + acq.GetPoint("RASI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"midASIS",valMidAsis,desc="")


        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acq,["LASI","LPSI","RASI","RPSI"])

        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,False)
        np.testing.assert_equal( globalFrame,"XYZ")


        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")


        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrame.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")

    @classmethod
    def gaitTrialProgressionY_backward_lateralX_static(cls):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\progression\\"


        gaitFilename="static_Y_backward.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        valMidAsis=(acq.GetPoint("LASI").GetValues() + acq.GetPoint("RASI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"midASIS",valMidAsis,desc="")


        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acq,["LASI","LPSI","RASI","RPSI"])

        np.testing.assert_equal( longitudinalAxis,"Y")
        np.testing.assert_equal( forwardProgression,False)
        np.testing.assert_equal( globalFrame,"YXZ")

        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"Y")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"YXZ")


        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrame.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"Y")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"YXZ")


    @classmethod
    def upperBody_StaticProgressionX_forward_lateralY(cls):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\progression\\"


        gaitFilename="fullBody_StaticX.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgression(acq,"CLAV")

        np.testing.assert_equal( longitudinalAxis,"X")
        np.testing.assert_equal( forwardProgression,True)
        np.testing.assert_equal( globalFrame,"XYZ")


        pfp = progressionFrame.ThoraxProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")


class IssueReported():

    @classmethod
    def brian(cls):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "\\Datasets Tests\\Brian Horsak\\issue-pelvis\\"


        gaitFilename="walk09.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        valSACR=acq.GetPoint("SACR").GetValues()

        btkTools.smartAppendPoint(acq,"RPSI",valSACR,desc="")
        btkTools.smartAppendPoint(acq,"LPSI",valSACR,desc="")


        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)


        gaitFilename="walk11.c3d"
        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        valSACR=acq.GetPoint("SACR").GetValues()

        btkTools.smartAppendPoint(acq,"RPSI",valSACR,desc="")
        btkTools.smartAppendPoint(acq,"LPSI",valSACR,desc="")


        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)


    @classmethod
    def gaitTrialGarches(cls):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\progression\\"

        translators = {
                "LASI":"L.ASIS",
                "RASI":"R.ASIS",
                "LPSI":"L.PSIS",
                "RPSI":"R.PSIS",
                "RTHI":"R.Thigh",
                "RKNE":"R.Knee",
                "RTHAP":"R.THAP",
                "RTHAD":"R.THAD",
                "RTIB":"R.Shank",
                "RANK":"R.Ankle",
                "RTIAP":"R.TIAP",
                "RTIAD":"R.TIAD",
                "RHEE":"R.Heel",
                "RSMH":"R.SMH",
                "RTOE":"R.Toe",
                "RFMH":"R.FMH",
                "RVMH":"R.VMH",
                "LTHI":"L.Thigh",
                "LKNE":"L.Knee",
                "LTHAP":"L.THAP",
                "LTHAD":"L.THAD",
                "LTIB":"L.Shank",
                "LANK":"L.Ankle",
                "LTIAP":"L.TIAP",
                "LTIAD":"L.TIAD",
                "LHEE":"L.Heel",
                "LSMH":"L.SMH",
                "LTOE":"L.Toe",
                "LFMH":"L.FMH",
                "LVMH":"L.VMH",
                "RKNM":"R.Knee.Medial",
                "LKNM":"L.Knee.Medial",
                "RMED":"R.Ankle.Medial",
                "LMED":"L.Ankle.Medial"
                }

        gaitFilename="gait_garches_issue.c3d"

        acq = btkTools.smartReader(str(MAIN_PATH +  gaitFilename),translators =translators )


        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"Y")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)





if __name__ == "__main__":

    # BtkProgressionTest_gaitTrial.gaitTrialProgressionX_forward_lateralY()
    # BtkProgressionTest_gaitTrial.gaitTrialProgressionX_backward_lateralY()
    # BtkProgressionTest_gaitTrial.gaitTrialProgressionY_forward_lateralX()
    # BtkProgressionTest_gaitTrial.gaitTrialProgressionY_backward_lateralX()

    # BtkProgressionTest_gaitTrial.upperBody_gaitTrialProgressionX_forward_lateralY()
    # BtkProgressionTest_gaitTrial.upperBody_gaitTrialProgressionX_backward_lateralY()


    # BtkProgressionTest_static.gaitTrialProgressionX_forward_lateralY_static()
    # BtkProgressionTest_static.gaitTrialProgressionX_backward_lateralY_static()
    # BtkProgressionTest_static.gaitTrialProgressionY_backward_lateralX_static()
    # BtkProgressionTest_static.upperBody_StaticProgressionX_forward_lateralY()


    # IssueReported.brian()
    IssueReported.gaitTrialGarches()
