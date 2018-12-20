# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:46:40 2016

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import numpy as np
import scipy as sp

import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2 import enums



class CGM1():


    @classmethod
    def CGM1_UpperLimb(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\full-PiG\\"
        staticFilename = "PN01NORMSTAT.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure(bodyPart=enums.BodyPart.UpperLimb)


        markerDiameter=14
        mp={
        'LeftShoulderOffset'   : 50,
        'LeftElbowWidth' : 91,
        'LeftWristWidth' : 56 ,
        'LeftHandThickness' : 28 ,
        'RightShoulderOffset'   : 45,
        'RightElbowWidth' : 90,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 30         }
        model.addAnthropoInputParameters(mp)

         # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqStatic)
        csdf.setStatic(True)
        csdf.display()

        btkTools.smartWriter(acqStatic,"upperLimb_calib.c3d")

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("TRXO").GetValues().mean(axis=0),acqStatic.GetPoint("OT").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LCLO").GetValues().mean(axis=0),acqStatic.GetPoint("LSJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("LHUO").GetValues().mean(axis=0),acqStatic.GetPoint("LEJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("LCLO").GetValues().mean(axis=0),acqStatic.GetPoint("LSJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("RCLO").GetValues().mean(axis=0),acqStatic.GetPoint("RSJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RHUO").GetValues().mean(axis=0),acqStatic.GetPoint("REJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RCLO").GetValues().mean(axis=0),acqStatic.GetPoint("RSJC").GetValues().mean(axis=0),decimal = 3)



    @classmethod
    def CGM1_fullbody(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\full-PiG\\"
        staticFilename = "PN01NORMSTAT-COM.c3d"

        # CALIBRATION ###############################
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        markerDiameter=14

        # Lower Limb
        mp={
        'Bodymass'   : 83,
        'LeftLegLength' : 874,
        'RightLegLength' : 876.0 ,
        'LeftKneeWidth' : 106.0,
        'RightKneeWidth' : 103.0,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 72.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 50,
        'LeftElbowWidth' : 91,
        'LeftWristWidth' : 56 ,
        'LeftHandThickness' : 28 ,
        'RightShoulderOffset'   : 45,
        'RightElbowWidth' : 90,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 30}

        model=cgm.CGM1LowerLimbs()
        model.configure(bodyPart=enums.BodyPart.FullBody)
        model.addAnthropoInputParameters(mp)

        scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = True,
                                            rightFlatFoot = True,
                                            markerDiameter = 14,
                                            viconCGM1compatible=True,
                                            ).compute()
                                            #headHorizontal=True

        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqStatic)
        csdf.setStatic(True)
        csdf.display()

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("TRXO").GetValues().mean(axis=0),acqStatic.GetPoint("OT").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LCLO").GetValues().mean(axis=0),acqStatic.GetPoint("LSJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("LHUO").GetValues().mean(axis=0),acqStatic.GetPoint("LEJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("LCLO").GetValues().mean(axis=0),acqStatic.GetPoint("LSJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("RCLO").GetValues().mean(axis=0),acqStatic.GetPoint("RSJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RHUO").GetValues().mean(axis=0),acqStatic.GetPoint("REJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RCLO").GetValues().mean(axis=0),acqStatic.GetPoint("RSJC").GetValues().mean(axis=0),decimal = 3)


        btkTools.smartWriter(acqStatic, "testFullStatic.c3d")

if __name__ == "__main__":



    CGM1.CGM1_UpperLimb()
    CGM1.CGM1_fullbody()

    logging.info("######## PROCESS CGM1 --> Done ######")
