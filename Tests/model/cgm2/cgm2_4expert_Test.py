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


if __name__ == "__main__":



    MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm 2_4\\bothSide\\"
    staticFilename = "static.c3d"
    gaitFilename= "gait 01.c3d"
    markerDiameter=14
    mp={
    'Bodymass'   : 64.5,
    'LeftLegLength' : 893.0,
    'RightLegLength' : 895.0 ,
    'LeftKneeWidth' : 91.0,
    'RightKneeWidth' : 89.0,
    'LeftAnkleWidth' : 66.0,
    'RightAnkleWidth' : 65.0,
    'LeftSoleDelta' : 0,
    'RightSoleDelta' : 0,
    "LeftToeOffset" : 0,
    "RightToeOffset" : 0,
    }

    CONTENT_INPUTS_CGM2_4 ="""
        {
        "Translators" : {
            "LASI":"",
            "RASI":"",
            "LPSI":"",
            "RPSI":"",
            "RTHI":"",
            "RKNE":"",
            "RTHIAP":"",
            "RTHIAD":"",
            "RTIB":"",
            "RANK":"",
            "RTIBAP":"RTIAP",
            "RTIBAD":"RSHN",
            "RHEE":"",
            "RTOE":"",
            "RCUN":"",
            "RD1M":"",
            "RD5M":"",
            "LTHI":"",
            "LKNE":"",
            "LTHIAP":"",
            "LTHIAD":"",
            "LTIB":"",
            "LANK":"",
            "LTIBAP":"LTIAP",
            "LTIBAD":"LSHN",
            "LHEE":"",
            "LTOE":"",
            "LCUN":"",
            "LD1M":"",
            "LD5M":""
            }
        }
      """


    # --- Calibration ---
    acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

    model=cgm2.CGM2_4LowerLimbs()
    model.configure()

    inputs = json.loads(CONTENT_INPUTS_CGM2_4,object_pairs_hook=OrderedDict)
    translators = inputs["Translators"]


    acqStatic =  btkTools.applyTranslators(acqStatic,translators)

    btkTools.smartWriter(acqStatic, "calibration2.c3d")

    model.addAnthropoInputParameters(mp)

    scp=modelFilters.StaticCalibrationProcedure(model)
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


#    # cgm decorator
    modelDecorator.HipJointCenterDecorator(model).hara()
#
#    # final
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model, useLeftHJCnode="LHJC_Hara", useRightHJCnode="RHJC_Hara").compute()

    btkTools.smartWriter(acqStatic, "calibration.c3d")

    # ------ Fitting -------
    acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

    acqGait =  btkTools.applyTranslators(acqGait,translators)
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


    btkTools.smartWriter(acqGait, "fitting-cgm2_4e.c3d")


    # ---Marker decomp filter----
    mtf = modelFilters.TrackingMarkerDecompositionFilter(model,acqGait)
    mtf.decompose()

    btkTools.smartWriter(acqGait, "fitting-cgm2_4e-decompose.c3d")

    # ------- OPENSIM IK --------------------------------------

    # --- osim builder ---
    cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model)
    markersetFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-markerset - expert.xml"

    osimfile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"


    oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                            model,
                                            cgmCalibrationprocedure)
    oscf.addMarkerSet(markersetFile)
    scalingOsim = oscf.build()
    oscf.exportXml("OSIMTEST.osim")


    # --- fitting ---

    #procedure

    cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model,expertMode = True)
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
#    cgmFittingProcedure.updateMarkerWeight("RCUN",100)
#    cgmFittingProcedure.updateMarkerWeight("RD1M",100)
#    cgmFittingProcedure.updateMarkerWeight("RD5M",100)
#    cgmFittingProcedure.updateMarkerWeight("RTOE",0)
#
#    cgmFittingProcedure.updateMarkerWeight("LTHI",100)
#    cgmFittingProcedure.updateMarkerWeight("LKNE",100)
#    cgmFittingProcedure.updateMarkerWeight("LTIB",100)
#    cgmFittingProcedure.updateMarkerWeight("LANK",100)
#    cgmFittingProcedure.updateMarkerWeight("LHEE",100)
#    cgmFittingProcedure.updateMarkerWeight("LCUN",100)
#    cgmFittingProcedure.updateMarkerWeight("LD1M",100)
#    cgmFittingProcedure.updateMarkerWeight("LD5M",100)
#    cgmFittingProcedure.updateMarkerWeight("LTOE",0)

    iksetupFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-expert-ikSetUp_template.xml"

    osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                      scalingOsim,
                                                      cgmFittingProcedure,
                                                      MAIN_PATH )


    acqIK = osrf.run(acqGait,str(MAIN_PATH + gaitFilename ))

    btkTools.smartWriter(acqIK,"fitting-cgm2_4e.c3d")




    # -------- NEW MOTION FILTER ON IK MARKERS ------------------

    modMotion_ik=modelFilters.ModelMotionFilter(scp,acqIK,model,pyCGM2Enums.motionMethod.Sodervisk,
                                                useForMotionTest=True)
    modMotion_ik.compute()

    finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
    finalJcs.setFilterBool(False)
    finalJcs.compute(description="ik", pointLabelSuffix = "2_ik")#



    btkTools.smartWriter(acqIK,"fitting-cgm2_4e-angles.c3d")
