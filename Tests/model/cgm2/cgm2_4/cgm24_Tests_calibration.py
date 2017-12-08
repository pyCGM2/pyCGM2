# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

import logging

import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.DEBUG)

import pyCGM2
# btk
pyCGM2.CONFIG.addBtk()

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm2
from pyCGM2.Model import  modelFilters,modelDecorator
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import numeric
from pyCGM2.Model.Opensim import opensimFilters
import json
from collections import OrderedDict


class CGM2_4_calibrationTest():


    @classmethod
    def calibration_noFlatFoot(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.4\\medial\\"
        staticFilename = "static.c3d"

        markerDiameter=14
        mp={
        'Bodymass'   : 69.0,
        'LeftLegLength' : 930.0,
        'RightLegLength' : 930.0 ,
        'LeftKneeWidth' : 94.0,
        'RightKneeWidth' : 64.0,
        'LeftAnkleWidth' : 67.0,
        'RightAnkleWidth' : 62.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        "LeftToeOffset" : 0,
        "RightToeOffset" : 0,
        }


        # --- Calibration ---
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm2.CGM2_4LowerLimbs()
        model.configure()

        model.addAnthropoInputParameters(mp)

        # ---- Calibration ----

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           markerDiameter=markerDiameter,
                           useDisplayPyCGM2_CoordinateSystem=True).compute()

        btkTools.smartWriter(acqStatic,"cgm2.4_noFlatFoot.c3d")

    @classmethod
    def calibration_FlatFoot(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.4\\medial\\"
        staticFilename = "static.c3d"

        markerDiameter=14
        mp={
        'Bodymass'   : 69.0,
        'LeftLegLength' : 930.0,
        'RightLegLength' : 930.0 ,
        'LeftKneeWidth' : 94.0,
        'RightKneeWidth' : 64.0,
        'LeftAnkleWidth' : 67.0,
        'RightAnkleWidth' : 62.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        "LeftToeOffset" : 0,
        "RightToeOffset" : 0,
        }


        # --- Calibration ---
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm2.CGM2_4LowerLimbs()
        model.configure()

        model.addAnthropoInputParameters(mp)

        # ---- Calibration ----

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           markerDiameter=markerDiameter,
                           leftFlatFoot = True, rightFlatFoot = True,
                           useDisplayPyCGM2_CoordinateSystem=True).compute()

        btkTools.smartWriter(acqStatic,"cgm2.4_FlatFoot.c3d")

if __name__ == "__main__":

    #CGM2_4_calibrationTest.calibration_noFlatFoot()
    CGM2_4_calibrationTest.calibration_FlatFoot()
