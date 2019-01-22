# -*- coding: utf-8 -*-
#import ipdb
import logging
import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model import modelFilters, modelDecorator,bodySegmentParameters
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools
from pyCGM2.Utils import files
from pyCGM2.Model.Opensim import opensimFilters


if __name__ == "__main__":

    plt.close("all")


    MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM2\\cgm2.2\\medial\\"
    staticFilename = "static-all.c3d"
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


    # --- Calibration ---
    acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

    model=cgm2.CGM2_2()
    model.configure()

    model.addAnthropoInputParameters(mp)

    # ------ calibration -------
    scp=modelFilters.StaticCalibrationProcedure(model)
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

    # cgm decorator
    modelDecorator.HipJointCenterDecorator(model).hara()
    modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
    modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

    # final
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
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
    markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm1\\cgm1-markerset.xml"

    osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"


    oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                            model,
                                            cgmCalibrationprocedure)
    oscf.addMarkerSet(markersetFile)
    scalingOsim = oscf.build(exportOsim=False)


    # --- fitting ---
    cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)
    iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm1\\cgm1-ikSetUp_template.xml"

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


    #btkTools.smartWriter(acqIK,"fitting-cgm2_2-angles.c3d")
