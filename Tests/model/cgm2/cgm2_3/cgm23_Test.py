# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pdb
import logging

import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

import pyCGM2
# btk
pyCGM2.CONFIG.addBtk()

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2.Model.CGM2 import cgm2
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import numeric
from pyCGM2.Model.Opensim import opensimFilters
import json
from collections import OrderedDict


if __name__ == "__main__":

    MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.3\\medial\\"
    staticFilename = "static.c3d"
    gaitFilename= "gait Trial 01.c3d"
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



    # --- Calibration ---
    acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

    model=cgm2.CGM2_3LowerLimbs()
    model.configure()

    model.addAnthropoInputParameters(mp)

    # ------ Calibration -------
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
    markersetFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-markerset.xml"

    osimfile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"


    oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                            model,
                                            cgmCalibrationprocedure)
    oscf.addMarkerSet(markersetFile)
    scalingOsim = oscf.build(exportOsim=False)


    # --- fitting ---
    #procedure
    cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)
#    cgmFittingProcedure.updateMarkerWeight("LASI",100)
#    cgmFittingProcedure.updateMarkerWeight("RASI",100)
#    cgmFittingProcedure.updateMarkerWeight("LPSI",100)
#    cgmFittingProcedure.updateMarkerWeight("RPSI",100)
#
#    cgmFittingProcedure.updateMarkerWeight("RTHI",100)
#    cgmFittingProcedure.updateMarkerWeight("RKNE",100)
#    cgmFittingProcedure.updateMarkerWeight("RTIB",100)
#    cgmFittingProcedure.updateMarkerWeight("RANK",100)
#    cgmFittingProcedure.updateMarkerWeight("RHEE",100)
#    cgmFittingProcedure.updateMarkerWeight("RTOE",100)
#    cgmFittingProcedure.updateMarkerWeight("LTHI",100)
#    cgmFittingProcedure.updateMarkerWeight("LKNE",100)
#    cgmFittingProcedure.updateMarkerWeight("LTIB",100)
#    cgmFittingProcedure.updateMarkerWeight("LANK",100)
#    cgmFittingProcedure.updateMarkerWeight("LHEE",100)
#    cgmFittingProcedure.updateMarkerWeight("LTOE",100)


    iksetupFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-ikSetUp_template.xml"

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



    btkTools.smartWriter(acqIK,"fitting-cgm2_3-angles.c3d")
