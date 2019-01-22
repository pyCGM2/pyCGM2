# -*- coding: utf-8 -*-
import numpy as np
import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2.Model.CGM2 import cgm

from pyCGM2 import enums




if __name__ == "__main__":



    MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\PIG advanced\\dynaKAD\\"
    staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

    acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

    model=cgm.CGM1
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

    #----CALIBRATION----

    # initial calibration
    scp=modelFilters.StaticCalibrationProcedure(model)
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


    # motion
    gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
    acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist,
                                             usePyCGM2_coordinateSystem=True)
    modMotion.compute()

    # calibration decorators
    modelDecorator.KneeCalibrationDecorator(model).calibrate2dof("Left")
    modelDecorator.KneeCalibrationDecorator(model).calibrate2dof("Right")

    # final calibration
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                        RotateLeftThigh = True,
                                        RotateRightThigh = True).compute()

    #----MOTION----
    # motion with dynakadOffset
    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist,
                                             usePyCGM2_coordinateSystem=True)
    modMotion.compute()

    btkTools.smartWriter(acqGait, "test2dof.c3d")
