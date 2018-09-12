# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import logging


import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)

import pyCGM2
# btk
from pyCGM2 import btk

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model import  modelFilters,modelDecorator
from pyCGM2.Model.CGM2 import cgm2
import pyCGM2.enums as pyCGM2Enums

from pyCGM2.Model.Opensim import opensimFilters


# enableLongitudinalRotation in Static and Motion filter rotate along Z
class CGM2_SARA_test():

    @classmethod
    def CGM2_3_SARA_test(cls):
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM2\\cgm2.3\\Knee Calibration\\"
        staticFilename = "Static.c3d"

        leftKneeFilename = "Left Knee.c3d"
        rightKneeFilename = "Right Knee.c3d"
        gaitFilename= "gait trial 01.c3d"


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


        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))


        model=cgm2.CGM2_3LowerLimbs()
        model.configure()

        model.addAnthropoInputParameters(mp)

        # --- INITIAL  CALIBRATION ---
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both",cgm1Behaviour=True)
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           seLeftHJCnode="LHJC_Hara", useRightHJCnode="RHJC_Hara",
                           useLeftKJCnode="LKJC_mid", useLeftAJCnode="LAJC_mid",
                           useRightKJCnode="RKJC_mid", useRightAJCnode="RAJC_mid",
                           markerDiameter=markerDiameter).compute()


        # ------ LEFT KNEE CALIBRATION -------
        acqLeftKnee = btkTools.smartReader(str(MAIN_PATH +  leftKneeFilename))

        # Motion of only left
        modMotionLeftKnee=modelFilters.ModelMotionFilter(scp,acqLeftKnee,model,pyCGM2Enums.motionMethod.Sodervisk)
        modMotionLeftKnee.segmentalCompute(["Left Thigh","Left Shank"])

        # decorator
        modelDecorator.KneeCalibrationDecorator(model).sara("Left",indexFirstFrame = 489,  indexLastFrame = 1451 )


        # ----add Point into the c3d----
        Or_inThigh = model.getSegment("Left Thigh").getReferential("TF").getNodeTrajectory("KneeFlexionOri")
        axis_inThigh = model.getSegment("Left Thigh").getReferential("TF").getNodeTrajectory("KneeFlexionAxis")
        btkTools.smartAppendPoint(acqLeftKnee,"Left" +"_KneeFlexionOri",Or_inThigh)
        btkTools.smartAppendPoint(acqLeftKnee,"Left" +"_KneeFlexionAxis",axis_inThigh)
        btkTools.smartWriter(acqLeftKnee, "Left Knee-Sara.c3d")

        # ------ RIGHT KNEE CALIBRATION -------
        acqRightKnee = btkTools.smartReader(str(MAIN_PATH +  rightKneeFilename))

        # Motion of only left
        modMotionRightKnee=modelFilters.ModelMotionFilter(scp,acqRightKnee,model,pyCGM2Enums.motionMethod.Sodervisk)
        modMotionRightKnee.segmentalCompute(["Right Thigh","Right Shank"])

        # decorator
        modelDecorator.KneeCalibrationDecorator(model).sara("Right",indexFirstFrame = 25,  indexLastFrame = 1060 )

        # ----add Point into the c3d----
        Or_inThigh = model.getSegment("Right Thigh").getReferential("TF").getNodeTrajectory("KneeFlexionOri")
        axis_inThigh = model.getSegment("Right Thigh").getReferential("TF").getNodeTrajectory("KneeFlexionAxis")
        btkTools.smartAppendPoint(acqRightKnee,"Right" +"_KneeFlexionOri",Or_inThigh)
        btkTools.smartAppendPoint(acqRightKnee,"Right" +"_KneeFlexionAxis",axis_inThigh)
        btkTools.smartWriter(acqRightKnee,  "Right Knee-Sara.c3d")

        #--- FINAL  CALIBRATION ---
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                        useLeftHJCnode="LHJC_Hara", useRightHJCnode="RHJC_Hara",
                        useLeftKJCnode="KJC_Sara", useLeftAJCnode="LAJC_mid",
                        useRightKJCnode="KJC_Sara", useRightAJCnode="RAJC_mid",
                        markerDiameter=markerDiameter,
                        RotateLeftThighFlag = True,
                        RotateRightThighFlag = True).compute()

        #  save static c3d with update KJC
        btkTools.smartWriter(acqStatic, "Static-SARA.c3d")


        # ------ Fitting -------
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))



        # Motion FILTER

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        # absolute angles
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","RASI","RPSI","LPSI"])
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "ROT"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")


        # ------- OPENSIM IK --------------------------------------

        # --- osim builder ---
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model)
        markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-markerset.xml"

        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"


        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure)
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build(exportOsim=False)


        # --- fitting ---
        #procedure
        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)
        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-ikSetUp_template.xml"

        osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          scalingOsim,
                                                          cgmFittingProcedure,
                                                          MAIN_PATH )
        acqIK = osrf.run(acqGait,str(MAIN_PATH + gaitFilename ),exportSetUp=False)

        # -------- NEW MOTION FILTER ON IK MARKERS ------------------

        modMotion_ik=modelFilters.ModelMotionFilter(scp,acqIK,model,pyCGM2Enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion_ik.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.setFilterBool(False)
        finalJcs.compute(description="ik", pointLabelSuffix = "2_ik")#

        btkTools.smartWriter(acqIK,"gait trial 01 - Fitting.c3d")



if __name__ == "__main__":

    CGM2_SARA_test.CGM2_3_SARA_test()
