# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:46:40 2016

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import numpy as np
import scipy as sp
import logging

import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# btk
pyCGM2.CONFIG.addBtk()

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Model import  modelFilters,modelDecorator, frame
import pyCGM2.enums as pyCGM2Enums

class CGM2_1_calibrationTest():


    @classmethod
    def hara_regressions(cls):
        """
        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.1\\native\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm2.CGM2_1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------
        # initial
        scp=modelFilters.StaticCalibrationProcedure(model)

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()
        pos0_L = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getLocal()
        pos0_R = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getLocal()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()
        pos_L = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getLocal()
        pos_R = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getLocal()

        # ---- tests ----
        np.testing.assert_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getDescription() ,"Hara")
        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LHJC").getDescription() ,"Hara")

        np.testing.assert_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getDescription() ,"Hara")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RHJC").getDescription() ,"Hara")

        np.testing.assert_equal(np.all(pos_L == pos0_L) == False, True)
        np.testing.assert_equal(np.all(pos_R == pos0_R) ==False, True)


    @classmethod
    def harrigton_fullPredictor(cls):
        """
        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.1\\native\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm2.CGM2_1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------
        # initial
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        pos0_L = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getLocal()
        pos0_R = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getLocal()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).harrington()

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


        pos_L = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getLocal()
        pos_R = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getLocal()

        # ---- tests ----
        np.testing.assert_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getDescription() ,"Harrington")
        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LHJC").getDescription() ,"Harrington")

        np.testing.assert_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getDescription() ,"Harrington")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RHJC").getDescription() ,"Harrington")

        np.testing.assert_equal(np.all(pos_L == pos0_L) == False, True)
        np.testing.assert_equal(np.all(pos_R == pos0_R) ==False, True)

    @classmethod
    def harrigton_pelvisWidthPredictor(cls):
        """
        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.1\\native\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm2.CGM2_1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------
        # initial
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        pos0_L = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getLocal()
        pos0_R = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getLocal()

         # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).harrington(predictors=pyCGM2Enums.HarringtonPredictor.PelvisWidth)

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


        pos_L = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getLocal()
        pos_R = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getLocal()

        # ---- tests ----
        np.testing.assert_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getDescription() ,"Harrington")
        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LHJC").getDescription() ,"Harrington")

        np.testing.assert_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getDescription() ,"Harrington")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RHJC").getDescription() ,"Harrington")

        np.testing.assert_equal(np.all(pos_L == pos0_L) == False, True)
        np.testing.assert_equal(np.all(pos_R == pos0_R) ==False, True)


    @classmethod
    def harrigton_legLengthPredictor(cls):
        """
        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.1\\native\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm2.CGM2_1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------
        # initial
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        pos0_L = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getLocal()
        pos0_R = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getLocal()

         # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).harrington(predictors=pyCGM2Enums.HarringtonPredictor.LegLength)

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


        pos_L = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getLocal()
        pos_R = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getLocal()

        # ---- tests ----
        np.testing.assert_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getDescription() ,"Harrington")
        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LHJC").getDescription() ,"Harrington")

        np.testing.assert_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getDescription() ,"Harrington")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RHJC").getDescription() ,"Harrington")

        np.testing.assert_equal(np.all(pos_L == pos0_L) == False, True)
        np.testing.assert_equal(np.all(pos_R == pos0_R) ==False, True)



    @classmethod
    def customLocalPosition(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.1\\native\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm2.CGM2_1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------
        # initial
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        pos0_L = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getLocal()
        pos0_R = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getLocal()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).custom(position_Left = np.array([1,2,3]),
                                                  position_Right = np.array([1,2,3]), methodDesc = "us") # add node to pelvis

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


        pos_L = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getLocal()
        pos_R = model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getLocal()

        # ---- tests ----
        np.testing.assert_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC").getDescription() ,"us")
        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LHJC").getDescription() ,"us")

        np.testing.assert_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC").getDescription() ,"us")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RHJC").getDescription() ,"us")

        np.testing.assert_equal(np.all(pos_L == pos0_L) == False, True)
        np.testing.assert_equal(np.all(pos_R == pos0_R) ==False, True)

if __name__ == "__main__":

    # CGM1 - custom
    logging.info("######## PROCESS  CGM2.1 ######")
    CGM2_1_calibrationTest.hara_regressions()
    CGM2_1_calibrationTest.harrigton_fullPredictor()
    CGM2_1_calibrationTest.harrigton_pelvisWidthPredictor()
    CGM2_1_calibrationTest.harrigton_legLengthPredictor()
    CGM2_1_calibrationTest.customLocalPosition()
    logging.info("######## PROCESS CGM2.1 --> Done ######")
