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
    def CGM2_4_SARA_test(cls):
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM2\\cgm2.4\\Knee Calibration\\"
        staticFilename = "static.c3d"

        funcFilename = "functional.c3d"
        gaitFilename= "gait trial 01.c3d"


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


        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))


        model=cgm2.CGM2_4LowerLimbs()
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
        acqFunc = btkTools.smartReader(str(MAIN_PATH +  funcFilename))

        # Motion of only left
        modMotionLeftKnee=modelFilters.ModelMotionFilter(scp,acqFunc,model,pyCGM2Enums.motionMethod.Sodervisk)
        modMotionLeftKnee.segmentalCompute(["Left Thigh","Left Shank"])

        # decorator
        modelDecorator.KneeCalibrationDecorator(model).sara("Left",indexFirstFrame = 831,  indexLastFrame = 1280 )


        # ----add Point into the c3d----
        Or_inThigh = model.getSegment("Left Thigh").getReferential("TF").getNodeTrajectory("KneeFlexionOri")
        axis_inThigh = model.getSegment("Left Thigh").getReferential("TF").getNodeTrajectory("KneeFlexionAxis")
        btkTools.smartAppendPoint(acqFunc,"Left" +"_KneeFlexionOri",Or_inThigh)
        btkTools.smartAppendPoint(acqFunc,"Left" +"_KneeFlexionAxis",axis_inThigh)


        # ------ RIGHT KNEE CALIBRATION -------

        # Motion of only left
        modMotionRightKnee=modelFilters.ModelMotionFilter(scp,acqFunc,model,pyCGM2Enums.motionMethod.Sodervisk)
        modMotionRightKnee.segmentalCompute(["Right Thigh","Right Shank"])

        # decorator
        modelDecorator.KneeCalibrationDecorator(model).sara("Right",indexFirstFrame = 61,  indexLastFrame = 551 )

        # ----add Point into the c3d----
        Or_inThigh = model.getSegment("Right Thigh").getReferential("TF").getNodeTrajectory("KneeFlexionOri")
        axis_inThigh = model.getSegment("Right Thigh").getReferential("TF").getNodeTrajectory("KneeFlexionAxis")
        btkTools.smartAppendPoint(acqFunc,"Right" +"_KneeFlexionOri",Or_inThigh)
        btkTools.smartAppendPoint(acqFunc,"Right" +"_KneeFlexionAxis",axis_inThigh)

        btkTools.smartWriter(acqFunc,  "acqFunc-Sara.c3d")

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
                                      segmentLabels=["Left HindFoot","Right HindFoot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "ROT"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")


        # ------- OPENSIM IK --------------------------------------

        # --- osim builder ---
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model)
        markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-markerset.xml"

        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"


        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure)
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build(exportOsim=False)


        # --- fitting ---
        #procedure
        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)
        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-ikSetUp_template.xml"

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

    CGM2_SARA_test.CGM2_4_SARA_test()
