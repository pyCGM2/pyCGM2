# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pdb
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

class CGM2_4_Tests():

    @classmethod
    def noIK(cls):
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.4\\medial\\"
        staticFilename = "static.c3d"
        gaitFilename= "gait Trial 01.c3d"

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
                                      segmentLabels=["Left HindFoot","Right HindFoot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "ROT"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")

    @classmethod
    def IK(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.4\\medial\\"
        staticFilename = "static.c3d"
        gaitFilename= "gait Trial 01.c3d"

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
                                      segmentLabels=["Left HindFoot","Right HindFoot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "ROT"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")


        # ------- OPENSIM IK --------------------------------------

        # --- osim builder ---
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model)
        markersetFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-markerset.xml"

        osimfile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"


        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure)
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build(exportOsim=False)

        # --- fitting ---
        #procedure
        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)

        iksetupFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-ikSetUp_template.xml"

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

        btkTools.smartWriter(acqIK,"fitting-cgm2_4-angles.c3d")


        #
        # # --- force plate handling----
        # # find foot  in contact
        # mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqIK)
        # forceplates.addForcePlateGeneralEvents(acqIK,mappedForcePlate)
        # logging.info("Force plate assignment : %s" %mappedForcePlate)
        #
        #
        # # assembly foot and force plate
        # modelFilters.ForcePlateAssemblyFilter(model,acqIK,mappedForcePlate,
        #                          leftSegmentLabel="Left HindFoot",
        #                          rightSegmentLabel="Right HindFoot").compute()
        #
        # #---- Joint kinetics----
        # idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        # modelFilters.InverseDynamicFilter(model,
        #                      acqIK,
        #                      procedure = idp,
        #                      projection = momentProjection
        #                      ).compute(pointLabelSuffix="ik")
        #
        # #---- Joint energetics----
        # modelFilters.JointPowerFilter(model,acqIK).compute(pointLabelSuffix="ik")
        #
        # btkTools.smartWriter(acqIK,"fitting-cgm2_4-kinetics.c3d")

if __name__ == "__main__":

    #CGM2_4_Tests.noIK()
    CGM2_4_Tests.IK()
