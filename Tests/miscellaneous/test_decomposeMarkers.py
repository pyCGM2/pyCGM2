# -*- coding: utf-8 -*-
import numpy as np
import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2.Model.CGM2 import cgm,cgm2
import pyCGM2.enums as pyCGM2Enums






class decomposeTrackingMarker_Test():

    @classmethod
    def cgm1_static(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\markerDecomposition\\CGM1decomposeTracking\\"
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

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
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
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\markerDecomposition\\CGM1decomposeTracking\\"
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

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 useForMotionTest=True)
        modMotion.compute()


        mtf = modelFilters.TrackingMarkerDecompositionFilter(model,acqGait)
        mtf.decompose()

        btkTools.smartWriter(acqGait, "cgm1-decompose.c3d")


    @classmethod
    def cgm24(cls):
        """

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\markerDecomposition\\CGM24decomposeTracking\\"
    #

        staticFilename = "PN01OP01S01STAT.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm2.CGM2_4LowerLimbs()
        model.setVersion("CGM2.4e")
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 83.0,
        'LeftLegLength' : 874.0,
        'RightLegLength' : 876.0 ,
        'LeftKneeWidth' : 106.0,
        'RightKneeWidth' : 103.0,
        'LeftAnkleWidth' : 74.0,
        'RightAnkleWidth' : 72.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # CALIBRATION
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = 1, rightFlatFoot = 1,
                                            markerDiameter=markerDiameter,
                                            ).compute()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="left")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="right")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")

        modelDecorator.HipJointCenterDecorator(model).hara(side = "Both")

        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = 1, rightFlatFoot = 1,
                                            markerDiameter=markerDiameter,
                                            ).compute()

        # --- Test 1 Motion Axe X -------
        gaitFilename="PN01OP01S01SS01.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Sodervisk                                             )
        modMotion.compute()


        mtf = modelFilters.TrackingMarkerDecompositionFilter(model,acqGait)
        mtf.decompose()

        btkTools.smartWriter(acqGait, "cgm24-decompose.c3d")



if __name__ == "__main__":

    logging.info("######## PROCESS CGM1 ######")
    decomposeTrackingMarker_Test.cgm1_static()
    decomposeTrackingMarker_Test.cgm1()
    logging.info("######## PROCESS CGM1 --> Done######")

    decomposeTrackingMarker_Test.cgm24()
