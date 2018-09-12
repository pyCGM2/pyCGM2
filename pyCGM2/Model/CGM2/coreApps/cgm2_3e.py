# -*- coding: utf-8 -*-
#import ipdb
import logging
import matplotlib.pyplot as plt
import argparse


# pyCGM2 settings
import pyCGM2

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2 import enums

from pyCGM2.Model import modelFilters, modelDecorator,bodySegmentParameters
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Model.CGM2.coreApps import cgmUtils
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Model.Opensim import opensimFilters



def calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
              required_mp,optional_mp,
              ik_flag,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethod,
              pointSuffix):

    # ---btk acquisition---
    acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))
    btkTools.checkMultipleSubject(acqStatic)

    acqStatic =  btkTools.applyTranslators(acqStatic,translators)



    # --------------------------MODEL--------------------------------------
    # ---definition---
    model=cgm2.CGM2_3LowerLimbs()
    model.setVersion("CGM2.3e")
    model.configure()
    model.addAnthropoInputParameters(required_mp,optional=optional_mp)

    # --store calibration parameters--
    model.setStaticFilename(calibrateFilenameLabelled)
    model.setCalibrationProperty("leftFlatFoot",leftFlatFoot)
    model.setCalibrationProperty("rightFlatFoot",rightFlatFoot)
    model.setCalibrationProperty("markerDiameter",markerDiameter)

    # ---check marker set used----
    smc= cgm.CGM.checkCGM1_StaticMarkerConfig(acqStatic)


    # --------------------------STATIC CALBRATION--------------------------
    scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

    # ---initial calibration filter----
    # use if all optional mp are zero
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                        leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                                        markerDiameter=markerDiameter,
                                        ).compute()

    # ---- Decorators -----
    # Goal = modified calibration according the identified marker set or if offsets manually set
    cgmUtils.applyDecorators_CGM(smc, model,acqStatic,optional_mp,markerDiameter)
    cgmUtils.applyHJCDecorators(model,hjcMethod)


    # ----Final Calibration filter if model previously decorated -----
    if model.decoratedModel:
        # initial static filter
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                           markerDiameter=markerDiameter).compute()



    # ----------------------CGM MODELLING----------------------------------
    # ----motion filter----
    modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,enums.motionMethod.Sodervisk,
                                              markerDiameter=markerDiameter)

    modMotion.compute()



    if ik_flag:

        # ---Marker decomp filter----
        mtf = modelFilters.TrackingMarkerDecompositionFilter(model,acqStatic)
        mtf.decompose()

        #                        ---OPENSIM IK---

        # --- opensim calibration Filter ---
        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
        markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-markerset - expert.xml" # markerset
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure,
                                                str(DATA_PATH))
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build()


        # --- opensim Fitting Filter ---
        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-expert-ikSetUp_template.xml" # ik tool file

        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model,expertMode = True) # procedure
        cgmFittingProcedure.updateMarkerWeight("LASI",settings["Fitting"]["Weight"]["LASI"])
        cgmFittingProcedure.updateMarkerWeight("RASI",settings["Fitting"]["Weight"]["RASI"])
        cgmFittingProcedure.updateMarkerWeight("LPSI",settings["Fitting"]["Weight"]["LPSI"])
        cgmFittingProcedure.updateMarkerWeight("RPSI",settings["Fitting"]["Weight"]["RPSI"])
        cgmFittingProcedure.updateMarkerWeight("RTHI",settings["Fitting"]["Weight"]["RTHI"])
        cgmFittingProcedure.updateMarkerWeight("RKNE",settings["Fitting"]["Weight"]["RKNE"])
        cgmFittingProcedure.updateMarkerWeight("RTIB",settings["Fitting"]["Weight"]["RTIB"])
        cgmFittingProcedure.updateMarkerWeight("RANK",settings["Fitting"]["Weight"]["RANK"])
        cgmFittingProcedure.updateMarkerWeight("RHEE",settings["Fitting"]["Weight"]["RHEE"])
        cgmFittingProcedure.updateMarkerWeight("RTOE",settings["Fitting"]["Weight"]["RTOE"])
        cgmFittingProcedure.updateMarkerWeight("LTHI",settings["Fitting"]["Weight"]["LTHI"])
        cgmFittingProcedure.updateMarkerWeight("LKNE",settings["Fitting"]["Weight"]["LKNE"])
        cgmFittingProcedure.updateMarkerWeight("LTIB",settings["Fitting"]["Weight"]["LTIB"])
        cgmFittingProcedure.updateMarkerWeight("LANK",settings["Fitting"]["Weight"]["LANK"])
        cgmFittingProcedure.updateMarkerWeight("LHEE",settings["Fitting"]["Weight"]["LHEE"])
        cgmFittingProcedure.updateMarkerWeight("LTOE",settings["Fitting"]["Weight"]["LTOE"])

        cgmFittingProcedure.updateMarkerWeight("LASI_posAnt",settings["Fitting"]["Weight"]["LASI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LASI_medLat",settings["Fitting"]["Weight"]["LASI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LASI_supInf",settings["Fitting"]["Weight"]["LASI_supInf"])

        cgmFittingProcedure.updateMarkerWeight("RASI_posAnt",settings["Fitting"]["Weight"]["RASI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RASI_medLat",settings["Fitting"]["Weight"]["RASI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RASI_supInf",settings["Fitting"]["Weight"]["RASI_supInf"])

        cgmFittingProcedure.updateMarkerWeight("LPSI_posAnt",settings["Fitting"]["Weight"]["LPSI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LPSI_medLat",settings["Fitting"]["Weight"]["LPSI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LPSI_supInf",settings["Fitting"]["Weight"]["LPSI_supInf"])

        cgmFittingProcedure.updateMarkerWeight("RPSI_posAnt",settings["Fitting"]["Weight"]["RPSI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RPSI_medLat",settings["Fitting"]["Weight"]["RPSI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RPSI_supInf",settings["Fitting"]["Weight"]["RPSI_supInf"])


        cgmFittingProcedure.updateMarkerWeight("RTHI_posAnt",settings["Fitting"]["Weight"]["RTHI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTHI_medLat",settings["Fitting"]["Weight"]["RTHI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTHI_proDis",settings["Fitting"]["Weight"]["RTHI_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RKNE_posAnt",settings["Fitting"]["Weight"]["RKNE_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RKNE_medLat",settings["Fitting"]["Weight"]["RKNE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RKNE_proDis",settings["Fitting"]["Weight"]["RKNE_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RTIB_posAnt",settings["Fitting"]["Weight"]["RTIB_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTIB_medLat",settings["Fitting"]["Weight"]["RTIB_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTIB_proDis",settings["Fitting"]["Weight"]["RTIB_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RANK_posAnt",settings["Fitting"]["Weight"]["RANK_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RANK_medLat",settings["Fitting"]["Weight"]["RANK_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RANK_proDis",settings["Fitting"]["Weight"]["RANK_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RHEE_supInf",settings["Fitting"]["Weight"]["RHEE_supInf"])
        cgmFittingProcedure.updateMarkerWeight("RHEE_medLat",settings["Fitting"]["Weight"]["RHEE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RHEE_proDis",settings["Fitting"]["Weight"]["RHEE_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RTOE_supInf",settings["Fitting"]["Weight"]["RTOE_supInf"])
        cgmFittingProcedure.updateMarkerWeight("RTOE_medLat",settings["Fitting"]["Weight"]["RTOE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTOE_proDis",settings["Fitting"]["Weight"]["RTOE_proDis"])



        cgmFittingProcedure.updateMarkerWeight("LTHI_posAnt",settings["Fitting"]["Weight"]["LTHI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTHI_medLat",settings["Fitting"]["Weight"]["LTHI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTHI_proDis",settings["Fitting"]["Weight"]["LTHI_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LKNE_posAnt",settings["Fitting"]["Weight"]["LKNE_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LKNE_medLat",settings["Fitting"]["Weight"]["LKNE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LKNE_proDis",settings["Fitting"]["Weight"]["LKNE_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LTIB_posAnt",settings["Fitting"]["Weight"]["LTIB_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTIB_medLat",settings["Fitting"]["Weight"]["LTIB_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTIB_proDis",settings["Fitting"]["Weight"]["LTIB_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LANK_posAnt",settings["Fitting"]["Weight"]["LANK_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LANK_medLat",settings["Fitting"]["Weight"]["LANK_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LANK_proDis",settings["Fitting"]["Weight"]["LANK_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LHEE_supInf",settings["Fitting"]["Weight"]["LHEE_supInf"])
        cgmFittingProcedure.updateMarkerWeight("LHEE_medLat",settings["Fitting"]["Weight"]["LHEE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LHEE_proDis",settings["Fitting"]["Weight"]["LHEE_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LTOE_supInf",settings["Fitting"]["Weight"]["LTOE_supInf"])
        cgmFittingProcedure.updateMarkerWeight("LTOE_medLat",settings["Fitting"]["Weight"]["LTOE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTOE_proDis",settings["Fitting"]["Weight"]["LTOE_proDis"])


        cgmFittingProcedure.updateMarkerWeight("LTHAP_posAnt",settings["Fitting"]["Weight"]["LTHAP_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTHAP_medLat",settings["Fitting"]["Weight"]["LTHAP_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTHAP_proDis",settings["Fitting"]["Weight"]["LTHAP_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LTHAD_posAnt",settings["Fitting"]["Weight"]["LTHAD_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTHAD_medLat",settings["Fitting"]["Weight"]["LTHAD_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTHAD_proDis",settings["Fitting"]["Weight"]["LTHAD_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RTHAP_posAnt",settings["Fitting"]["Weight"]["RTHAP_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTHAP_medLat",settings["Fitting"]["Weight"]["RTHAP_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTHAP_proDis",settings["Fitting"]["Weight"]["RTHAP_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RTHAD_posAnt",settings["Fitting"]["Weight"]["RTHAD_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTHAD_medLat",settings["Fitting"]["Weight"]["RTHAD_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTHAD_proDis",settings["Fitting"]["Weight"]["RTHAD_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LTIAP_posAnt",settings["Fitting"]["Weight"]["LTIAP_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTIAP_medLat",settings["Fitting"]["Weight"]["LTIAP_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTIAP_proDis",settings["Fitting"]["Weight"]["LTIAP_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LTIAD_posAnt",settings["Fitting"]["Weight"]["LTIAD_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTIAD_medLat",settings["Fitting"]["Weight"]["LTIAD_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTIAD_proDis",settings["Fitting"]["Weight"]["LTIAD_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RTIAP_posAnt",settings["Fitting"]["Weight"]["RTIAP_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTIAP_medLat",settings["Fitting"]["Weight"]["RTIAP_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTIAP_proDis",settings["Fitting"]["Weight"]["RTIAP_proDis"])


        cgmFittingProcedure.updateMarkerWeight("RTIAD_posAnt",settings["Fitting"]["Weight"]["RTIAD_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTIAD_medLat",settings["Fitting"]["Weight"]["RTIAD_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTIAD_proDis",settings["Fitting"]["Weight"]["RTIAD_proDis"])

#            cgmFittingProcedure.updateMarkerWeight("LPAT",settings["Fitting"]["Weight"]["LPAT"])
#            cgmFittingProcedure.updateMarkerWeight("LPAT_posAnt",settings["Fitting"]["Weight"]["LPAT_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("LPAT_medLat",settings["Fitting"]["Weight"]["LPAT_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("LPAT_proDis",settings["Fitting"]["Weight"]["LPAT_proDis"])
#
#            cgmFittingProcedure.updateMarkerWeight("RPAT",settings["Fitting"]["Weight"]["RPAT"])
#            cgmFittingProcedure.updateMarkerWeight("RPAT_posAnt",settings["Fitting"]["Weight"]["RPAT_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("RPAT_medLat",settings["Fitting"]["Weight"]["RPAT_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("RPAT_proDis",settings["Fitting"]["Weight"]["RPAT_proDis"])

#            cgmFittingProcedure.updateMarkerWeight("LTHLD",settings["Fitting"]["Weight"]["LTHLD"])
#            cgmFittingProcedure.updateMarkerWeight("LTHLD_posAnt",settings["Fitting"]["Weight"]["LTHLD_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("LTHLD_medLat",settings["Fitting"]["Weight"]["LTHLD_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("LTHLD_proDis",settings["Fitting"]["Weight"]["LTHLD_proDis"])
#
#            cgmFittingProcedure.updateMarkerWeight("RTHLD",settings["Fitting"]["Weight"]["RTHLD"])
#            cgmFittingProcedure.updateMarkerWeight("RTHLD_posAnt",settings["Fitting"]["Weight"]["RTHLD_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("RTHLD_medLat",settings["Fitting"]["Weight"]["RTHLD_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("RTHLD_proDis",settings["Fitting"]["Weight"]["RTHLD_proDis"])

        osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          scalingOsim,
                                                          cgmFittingProcedure,
                                                          str(DATA_PATH) )
        acqStaticIK = osrf.run(acqStatic,str(DATA_PATH + calibrateFilenameLabelled ))



    # eventual static acquisition to consider for joint kinematics
    finalAcqStatic = acqStaticIK if ik_flag else acqStatic

    # --- final pyCGM2 model motion Filter ---
    # use fitted markers
    modMotionFitted=modelFilters.ModelMotionFilter(scp,finalAcqStatic,model,enums.motionMethod.Sodervisk)
    modMotionFitted.compute()

    #---- Joint kinematics----
    # relative angles
    modelFilters.ModelJCSFilter(model,finalAcqStatic).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

    # detection of traveling axis
    longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(finalAcqStatic,["LASI","RASI","RPSI","LPSI"])

    # absolute angles
    modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqStatic,
                                           segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                            angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                            eulerSequences=["TOR","TOR", "ROT"],
                                            globalFrameOrientation = globalFrame,
                                            forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)



    return model, finalAcqStatic


def fitting(model,DATA_PATH, reconstructFilenameLabelled,
    translators,settings,
    ik_flag,markerDiameter,
    pointSuffix,
    mfpa,
    momentProjection):

    # --- btk acquisition ----
    acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

    btkTools.checkMultipleSubject(acqGait)
    acqGait =  btkTools.applyTranslators(acqGait,translators)
    validFrames,vff,vlf = btkTools.findValidFrames(acqGait,cgm2.CGM2_3LowerLimbs.MARKERS)


    # --- initial motion Filter ---
    scp=modelFilters.StaticCalibrationProcedure(model)
    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
    modMotion.compute()

    if ik_flag:
        #                        ---OPENSIM IK---

        # ---Marker decomp filter----
        mtf = modelFilters.TrackingMarkerDecompositionFilter(model,acqGait)
        mtf.decompose()


        # --- opensim calibration Filter ---
        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
        markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-markerset - expert.xml" # markerset
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure,
                                                str(DATA_PATH))
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build()

        # --- opensim Fitting Filter ---
        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-expert-ikSetUp_template.xml" # ik tl file

        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model,expertMode = True) # procedure
        cgmFittingProcedure.updateMarkerWeight("LASI",settings["Fitting"]["Weight"]["LASI"])
        cgmFittingProcedure.updateMarkerWeight("RASI",settings["Fitting"]["Weight"]["RASI"])
        cgmFittingProcedure.updateMarkerWeight("LPSI",settings["Fitting"]["Weight"]["LPSI"])
        cgmFittingProcedure.updateMarkerWeight("RPSI",settings["Fitting"]["Weight"]["RPSI"])
        cgmFittingProcedure.updateMarkerWeight("RTHI",settings["Fitting"]["Weight"]["RTHI"])
        cgmFittingProcedure.updateMarkerWeight("RKNE",settings["Fitting"]["Weight"]["RKNE"])
        cgmFittingProcedure.updateMarkerWeight("RTIB",settings["Fitting"]["Weight"]["RTIB"])
        cgmFittingProcedure.updateMarkerWeight("RANK",settings["Fitting"]["Weight"]["RANK"])
        cgmFittingProcedure.updateMarkerWeight("RHEE",settings["Fitting"]["Weight"]["RHEE"])
        cgmFittingProcedure.updateMarkerWeight("RTOE",settings["Fitting"]["Weight"]["RTOE"])
        cgmFittingProcedure.updateMarkerWeight("LTHI",settings["Fitting"]["Weight"]["LTHI"])
        cgmFittingProcedure.updateMarkerWeight("LKNE",settings["Fitting"]["Weight"]["LKNE"])
        cgmFittingProcedure.updateMarkerWeight("LTIB",settings["Fitting"]["Weight"]["LTIB"])
        cgmFittingProcedure.updateMarkerWeight("LANK",settings["Fitting"]["Weight"]["LANK"])
        cgmFittingProcedure.updateMarkerWeight("LHEE",settings["Fitting"]["Weight"]["LHEE"])
        cgmFittingProcedure.updateMarkerWeight("LTOE",settings["Fitting"]["Weight"]["LTOE"])

        cgmFittingProcedure.updateMarkerWeight("LASI_posAnt",settings["Fitting"]["Weight"]["LASI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LASI_medLat",settings["Fitting"]["Weight"]["LASI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LASI_supInf",settings["Fitting"]["Weight"]["LASI_supInf"])

        cgmFittingProcedure.updateMarkerWeight("RASI_posAnt",settings["Fitting"]["Weight"]["RASI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RASI_medLat",settings["Fitting"]["Weight"]["RASI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RASI_supInf",settings["Fitting"]["Weight"]["RASI_supInf"])

        cgmFittingProcedure.updateMarkerWeight("LPSI_posAnt",settings["Fitting"]["Weight"]["LPSI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LPSI_medLat",settings["Fitting"]["Weight"]["LPSI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LPSI_supInf",settings["Fitting"]["Weight"]["LPSI_supInf"])

        cgmFittingProcedure.updateMarkerWeight("RPSI_posAnt",settings["Fitting"]["Weight"]["RPSI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RPSI_medLat",settings["Fitting"]["Weight"]["RPSI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RPSI_supInf",settings["Fitting"]["Weight"]["RPSI_supInf"])


        cgmFittingProcedure.updateMarkerWeight("RTHI_posAnt",settings["Fitting"]["Weight"]["RTHI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTHI_medLat",settings["Fitting"]["Weight"]["RTHI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTHI_proDis",settings["Fitting"]["Weight"]["RTHI_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RKNE_posAnt",settings["Fitting"]["Weight"]["RKNE_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RKNE_medLat",settings["Fitting"]["Weight"]["RKNE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RKNE_proDis",settings["Fitting"]["Weight"]["RKNE_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RTIB_posAnt",settings["Fitting"]["Weight"]["RTIB_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTIB_medLat",settings["Fitting"]["Weight"]["RTIB_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTIB_proDis",settings["Fitting"]["Weight"]["RTIB_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RANK_posAnt",settings["Fitting"]["Weight"]["RANK_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RANK_medLat",settings["Fitting"]["Weight"]["RANK_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RANK_proDis",settings["Fitting"]["Weight"]["RANK_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RHEE_supInf",settings["Fitting"]["Weight"]["RHEE_supInf"])
        cgmFittingProcedure.updateMarkerWeight("RHEE_medLat",settings["Fitting"]["Weight"]["RHEE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RHEE_proDis",settings["Fitting"]["Weight"]["RHEE_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RTOE_supInf",settings["Fitting"]["Weight"]["RTOE_supInf"])
        cgmFittingProcedure.updateMarkerWeight("RTOE_medLat",settings["Fitting"]["Weight"]["RTOE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTOE_proDis",settings["Fitting"]["Weight"]["RTOE_proDis"])



        cgmFittingProcedure.updateMarkerWeight("LTHI_posAnt",settings["Fitting"]["Weight"]["LTHI_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTHI_medLat",settings["Fitting"]["Weight"]["LTHI_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTHI_proDis",settings["Fitting"]["Weight"]["LTHI_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LKNE_posAnt",settings["Fitting"]["Weight"]["LKNE_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LKNE_medLat",settings["Fitting"]["Weight"]["LKNE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LKNE_proDis",settings["Fitting"]["Weight"]["LKNE_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LTIB_posAnt",settings["Fitting"]["Weight"]["LTIB_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTIB_medLat",settings["Fitting"]["Weight"]["LTIB_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTIB_proDis",settings["Fitting"]["Weight"]["LTIB_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LANK_posAnt",settings["Fitting"]["Weight"]["LANK_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LANK_medLat",settings["Fitting"]["Weight"]["LANK_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LANK_proDis",settings["Fitting"]["Weight"]["LANK_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LHEE_supInf",settings["Fitting"]["Weight"]["LHEE_supInf"])
        cgmFittingProcedure.updateMarkerWeight("LHEE_medLat",settings["Fitting"]["Weight"]["LHEE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LHEE_proDis",settings["Fitting"]["Weight"]["LHEE_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LTOE_supInf",settings["Fitting"]["Weight"]["LTOE_supInf"])
        cgmFittingProcedure.updateMarkerWeight("LTOE_medLat",settings["Fitting"]["Weight"]["LTOE_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTOE_proDis",settings["Fitting"]["Weight"]["LTOE_proDis"])


        cgmFittingProcedure.updateMarkerWeight("LTHAP_posAnt",settings["Fitting"]["Weight"]["LTHAP_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTHAP_medLat",settings["Fitting"]["Weight"]["LTHAP_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTHAP_proDis",settings["Fitting"]["Weight"]["LTHAP_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LTHAD_posAnt",settings["Fitting"]["Weight"]["LTHAD_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTHAD_medLat",settings["Fitting"]["Weight"]["LTHAD_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTHAD_proDis",settings["Fitting"]["Weight"]["LTHAD_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RTHAP_posAnt",settings["Fitting"]["Weight"]["RTHAP_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTHAP_medLat",settings["Fitting"]["Weight"]["RTHAP_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTHAP_proDis",settings["Fitting"]["Weight"]["RTHAP_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RTHAD_posAnt",settings["Fitting"]["Weight"]["RTHAD_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTHAD_medLat",settings["Fitting"]["Weight"]["RTHAD_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTHAD_proDis",settings["Fitting"]["Weight"]["RTHAD_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LTIAP_posAnt",settings["Fitting"]["Weight"]["LTIAP_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTIAP_medLat",settings["Fitting"]["Weight"]["LTIAP_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTIAP_proDis",settings["Fitting"]["Weight"]["LTIAP_proDis"])

        cgmFittingProcedure.updateMarkerWeight("LTIAD_posAnt",settings["Fitting"]["Weight"]["LTIAD_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("LTIAD_medLat",settings["Fitting"]["Weight"]["LTIAD_medLat"])
        cgmFittingProcedure.updateMarkerWeight("LTIAD_proDis",settings["Fitting"]["Weight"]["LTIAD_proDis"])

        cgmFittingProcedure.updateMarkerWeight("RTIAP_posAnt",settings["Fitting"]["Weight"]["RTIAP_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTIAP_medLat",settings["Fitting"]["Weight"]["RTIAP_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTIAP_proDis",settings["Fitting"]["Weight"]["RTIAP_proDis"])


        cgmFittingProcedure.updateMarkerWeight("RTIAD_posAnt",settings["Fitting"]["Weight"]["RTIAD_posAnt"])
        cgmFittingProcedure.updateMarkerWeight("RTIAD_medLat",settings["Fitting"]["Weight"]["RTIAD_medLat"])
        cgmFittingProcedure.updateMarkerWeight("RTIAD_proDis",settings["Fitting"]["Weight"]["RTIAD_proDis"])

#            cgmFittingProcedure.updateMarkerWeight("LPAT",settings["Fitting"]["Weight"]["LPAT"])
#            cgmFittingProcedure.updateMarkerWeight("LPAT_posAnt",settings["Fitting"]["Weight"]["LPAT_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("LPAT_medLat",settings["Fitting"]["Weight"]["LPAT_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("LPAT_proDis",settings["Fitting"]["Weight"]["LPAT_proDis"])
#
#            cgmFittingProcedure.updateMarkerWeight("RPAT",settings["Fitting"]["Weight"]["RPAT"])
#            cgmFittingProcedure.updateMarkerWeight("RPAT_posAnt",settings["Fitting"]["Weight"]["RPAT_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("RPAT_medLat",settings["Fitting"]["Weight"]["RPAT_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("RPAT_proDis",settings["Fitting"]["Weight"]["RPAT_proDis"])
#
#            cgmFittingProcedure.updateMarkerWeight("LTHLD",settings["Fitting"]["Weight"]["LTHLD"])
#            cgmFittingProcedure.updateMarkerWeight("LTHLD_posAnt",settings["Fitting"]["Weight"]["LTHLD_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("LTHLD_medLat",settings["Fitting"]["Weight"]["LTHLD_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("LTHLD_proDis",settings["Fitting"]["Weight"]["LTHLD_proDis"])
#
#            cgmFittingProcedure.updateMarkerWeight("RTHLD",settings["Fitting"]["Weight"]["RTHLD"])
#            cgmFittingProcedure.updateMarkerWeight("RTHLD_posAnt",settings["Fitting"]["Weight"]["RTHLD_posAnt"])
#            cgmFittingProcedure.updateMarkerWeight("RTHLD_medLat",settings["Fitting"]["Weight"]["RTHLD_medLat"])
#            cgmFittingProcedure.updateMarkerWeight("RTHLD_proDis",settings["Fitting"]["Weight"]["RTHLD_proDis"])


        osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          scalingOsim,
                                                          cgmFittingProcedure,
                                                          str(DATA_PATH) )


        acqIK = osrf.run(acqGait,str(DATA_PATH + reconstructFilenameLabelled ))


    # eventual gait acquisition to consider for joint kinematics
    finalAcqGait = acqIK if ik_flag else acqGait

    # --- final pyCGM2 model motion Filter ---
    # use fitted markers
    modMotionFitted=modelFilters.ModelMotionFilter(scp,finalAcqGait,model,enums.motionMethod.Sodervisk ,
                                              markerDiameter=markerDiameter)

    modMotionFitted.compute()


    #---- Joint kinematics----
    # relative angles
    modelFilters.ModelJCSFilter(model,finalAcqGait).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

    # detection of traveling axis
    longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(finalAcqGait,["LASI","LPSI","RASI","RPSI"])


    # absolute angles
    modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqGait,
                                           segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                            angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                            eulerSequences=["TOR","TOR", "ROT"],
                                            globalFrameOrientation = globalFrame,
                                            forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

    #---- Body segment parameters----
    bspModel = bodySegmentParameters.Bsp(model)
    bspModel.compute()

    # --- force plate handling----
    # find foot  in contact
    mappedForcePlate = forceplates.matchingFootSideOnForceplate(finalAcqGait)
    forceplates.addForcePlateGeneralEvents(finalAcqGait,mappedForcePlate)
    logging.info("Force plate assignment : %s" %mappedForcePlate)

    if mfpa is not None:
        if len(mfpa) != len(mappedForcePlate):
            raise Exception("[pyCGM2] manual force plate assignment badly sets. Wrong force plate number. %s force plate require" %(str(len(mappedForcePlate))))
        else:
            mappedForcePlate = mfpa
            forceplates.addForcePlateGeneralEvents(finalAcqGait,mappedForcePlate)
            logging.warning("Manual Force plate assignment : %s" %mappedForcePlate)

    # assembly foot and force plate
    modelFilters.ForcePlateAssemblyFilter(model,finalAcqGait,mappedForcePlate,
                             leftSegmentLabel="Left Foot",
                             rightSegmentLabel="Right Foot").compute()

    #---- Joint kinetics----
    idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
    modelFilters.InverseDynamicFilter(model,
                         finalAcqGait,
                         procedure = idp,
                         projection = momentProjection
                         ).compute(pointLabelSuffix=pointSuffix)

    #---- Joint energetics----
    modelFilters.JointPowerFilter(model,finalAcqGait).compute(pointLabelSuffix=pointSuffix)

    #---- zero unvalid frames ---
    btkTools.applyValidFramesOnOutput(finalAcqGait,validFrames)



    return finalAcqGait
