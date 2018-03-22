# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

import pyCGM2
# btk
pyCGM2.CONFIG.addBtk()

# pyCGM2
from pyCGM2 import enums
from pyCGM2.Tools import  btkTools
from pyCGM2.Model import  modelFilters,modelDecorator, frame, modelQualityFilter
from pyCGM2.Model.CGM2 import cgm




class CGM1():

    @classmethod
    def ModelScoreTest(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        # score

        msrp = modelQualityFilter.ModelScoreResidualProcedure(model)
        srf = modelQualityFilter.ScoreResidualFilter(acqGait,msrp)
        srf.compute()

    @classmethod
    def GeneralScoreTest(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
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

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        # score

        msrp = modelQualityFilter.GeneralScoreResidualProcedure(model)
        msrp.setDefinition("LHJC","Pelvis","Left Thigh")

        srf = modelQualityFilter.ScoreResidualFilter(acqGait,msrp)
        srf.compute()


if __name__ == "__main__":

    #    plt.close("all")

    CGM1.ModelScoreTest()
    CGM1.GeneralScoreTest()
