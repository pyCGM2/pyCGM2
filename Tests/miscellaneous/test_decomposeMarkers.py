# -*- coding: utf-8 -*-
import numpy as np
import pdb
import logging

import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# btk
pyCGM2.CONFIG.addBtk()

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2.Model.CGM2 import cgm
import pyCGM2.enums as pyCGM2Enums






class decomposeTrackingMarker_Test():

    @classmethod
    def cgm1_static(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\decomposeTracking\\"
#

        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 36.9,
        'LeftLegLength' : 665,
        'RightLegLength' : 655.0 ,
        'LeftKneeWidth' : 102.7,
        'RightKneeWidth' : 100.2,
        'LeftAnkleWidth' : 64.5,
        'RightAnkleWidth' : 63.4,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # CALIBRATION
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        print model.m_useRightTibialTorsion

        # --- Test 1 Motion Axe X -------
        gaitFilename="static.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native,
                                                 useForMotionTest=True)
        modMotion.compute()



        mtf = modelFilters.TrackingMarkerDecompositionFilter(model,acqGait)
        mtf.decompose()

        btkTools.smartWriter(acqGait, "cgm1_static.c3d")



    @classmethod
    def cgm1(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\decomposeTracking\\"
#

        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 36.9,
        'LeftLegLength' : 665,
        'RightLegLength' : 655.0 ,
        'LeftKneeWidth' : 102.7,
        'RightKneeWidth' : 100.2,
        'LeftAnkleWidth' : 64.5,
        'RightAnkleWidth' : 63.4,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # CALIBRATION
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        print model.m_useRightTibialTorsion

        # --- Test 1 Motion Axe X -------
        gaitFilename="gait trial 01.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native,
                                                 useForMotionTest=True)
        modMotion.compute()


        mtf = modelFilters.TrackingMarkerDecompositionFilter(model,acqGait)
        mtf.decompose()

        btkTools.smartWriter(acqGait, "cgm1-decompose.c3d")


if __name__ == "__main__":

    logging.info("######## PROCESS CGM1 ######")
    decomposeTrackingMarker_Test.cgm1_static()
    decomposeTrackingMarker_Test.cgm1()
    logging.info("######## PROCESS CGM1 --> Done######")
