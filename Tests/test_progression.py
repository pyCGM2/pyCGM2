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
import pyCGM2; LOGGER = pyCGM2.LOGGER

from pyCGM2.Tools import  btkTools
from pyCGM2.Processing.ProgressionFrame import progressionFrameFilters
from pyCGM2.Processing.ProgressionFrame import progressionFrameProcedures
from pyCGM2.Utils import utils
from pyCGM2.Lib.Processing import progression

class Test_LibProgression():

    def test_gaitTrialProgression(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="gait_X_forward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)
        progressionAxis, forwardProgression, globalFrame = progression.detectProgressionFrame(acq, staticFlag=False)
        np.testing.assert_equal( progressionAxis,"X")
        np.testing.assert_equal( forwardProgression ,True)
        np.testing.assert_equal( globalFrame,"XYZ")

        gaitFilename="gait_X_backward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)
        progressionAxis, forwardProgression, globalFrame = progression.detectProgressionFrame(acq, staticFlag=False)
        np.testing.assert_equal( progressionAxis,"X")
        np.testing.assert_equal( forwardProgression ,False)
        np.testing.assert_equal( globalFrame,"XYZ")


        gaitFilename="gait_Y_forward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)
        progressionAxis, forwardProgression, globalFrame = progression.detectProgressionFrame(acq, staticFlag=False)
        np.testing.assert_equal( progressionAxis,"Y")
        np.testing.assert_equal( forwardProgression ,True)
        np.testing.assert_equal( globalFrame,"YXZ")

        gaitFilename="gait_Y_backward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)
        progressionAxis, forwardProgression, globalFrame = progression.detectProgressionFrame(acq, staticFlag=False)
        np.testing.assert_equal( progressionAxis,"Y")
        np.testing.assert_equal( forwardProgression ,False)
        np.testing.assert_equal( globalFrame,"YXZ")


    def test_staticTrialProgression(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="static_X.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)
        progressionAxis, forwardProgression, globalFrame = progression.detectProgressionFrame(acq, staticFlag=True)
        np.testing.assert_equal( progressionAxis,"X")
        np.testing.assert_equal( forwardProgression ,True)
        np.testing.assert_equal( globalFrame,"XYZ")

        gaitFilename="static_X_backward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)
        progressionAxis, forwardProgression, globalFrame = progression.detectProgressionFrame(acq, staticFlag=True)
        np.testing.assert_equal( progressionAxis,"X")
        np.testing.assert_equal( forwardProgression ,False)
        np.testing.assert_equal( globalFrame,"XYZ")


        gaitFilename="upperBody_StaticX.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)
        progressionAxis, forwardProgression, globalFrame = progression.detectProgressionFrame(acq, staticFlag=True)
        np.testing.assert_equal( progressionAxis,"X")
        np.testing.assert_equal( forwardProgression ,True)
        np.testing.assert_equal( globalFrame,"XYZ")


        gaitFilename="static_Y_backward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)
        progressionAxis, forwardProgression, globalFrame = progression.detectProgressionFrame(acq, staticFlag=True)
        np.testing.assert_equal( progressionAxis,"Y")
        np.testing.assert_equal( forwardProgression ,False)
        np.testing.assert_equal( globalFrame,"YXZ")



# Gait
class Test_btkProgression():

    def test_gaitTrialProgressionX_forward_lateralY(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="gait_X_forward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")


        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
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


        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")


        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
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

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"Y")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"YXZ")

        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
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


        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"Y")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"YXZ")

        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
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

        pfp = progressionFrameProcedures.ThoraxProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")



    def test_upperBody_gaitTrialProgressionX_backward_lateralY(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="UpperBody_GaitX_backward.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

        pfp = progressionFrameProcedures.ThoraxProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
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

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")


        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
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

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")


        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
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

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"Y")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"YXZ")


        valSACR=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acq,"SACR",valSACR,desc="")

        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure(backMarkers=["SACR"])
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"Y")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,False)
        np.testing.assert_equal( pff.outputs["globalFrame"],"YXZ")



    def test_upperBody_StaticProgressionX_forward_lateralY(self):
        """
        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\ProgressionFrame\\sample 1\\"


        gaitFilename="upperBody_StaticX.c3d"
        acq = btkTools.smartReader(MAIN_PATH +  gaitFilename)

        pfp = progressionFrameProcedures.ThoraxProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        np.testing.assert_equal( pff.outputs["progressionAxis"],"X")
        np.testing.assert_equal( pff.outputs["forwardProgression"] ,True)
        np.testing.assert_equal( pff.outputs["globalFrame"],"XYZ")
