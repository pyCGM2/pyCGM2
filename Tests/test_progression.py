# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_progression.py::Test_btkProgression
"""
Created on Thu Sep 15 11:09:22 2016

@author: aaa34169

TODO : these cases are lacking :
 - progression Z lateral axis (X or Y)
 - progression X lateral axis Z
 - progression Y lateral axis Z

"""

import numpy as np
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2.Tools import  btkTools
from pyCGM2.Processing import progressionFrame
from pyCGM2.Utils import utils
# ---- BTK ------

# Gait
class Test_btkProgression():

    def test_gaitTrialProgressionX_forward_lateralY(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="gait_X_forward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

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


    def test_gaitTrialProgressionX_backward_lateralY(self):
        """

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"

        gaitFilename="gait_X_backward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")


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



    def test_gaitTrialProgressionY_forward_lateralX(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="gait_Y_forward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

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




    def test_gaitTrialProgressionY_backward_lateralX(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="gait_Y_backward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)


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




    def test_upperBody_gaitTrialProgressionX_forward_lateralY(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="fullBody_GaitX_forward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

        pfp = progressionFrame.ThoraxProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")



    def test_upperBody_gaitTrialProgressionX_backward_lateralY(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="fullBody_GaitX_backward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

        pfp = progressionFrame.ThoraxProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")

#--- static
class Test_btkProgression_static():


    def test_gaitTrialProgressionX_forward_lateralY_static(self):
        """

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"

        gaitFilename="static_X.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

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





    def test_gaitTrialProgressionX_backward_lateralY_static(self):
        """

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"

        gaitFilename="static_X_backward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

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


    def test_gaitTrialProgressionY_backward_lateralX_static(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="static_Y_backward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

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



    def test_upperBody_StaticProgressionX_forward_lateralY(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="fullBody_StaticX.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

        pfp = progressionFrame.ThoraxProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")
