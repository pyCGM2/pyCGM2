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
from pyCGM2.Model.CGM2.coreApps import decorators
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

    validFrames,vff,vlf = btkTools.findValidFrames(acqStatic,cgm2.CGM2_4LowerLimbs.TRACKING_MARKERS) 

    # --------------------------MODEL--------------------------------------
    # ---definition---
    model=cgm2.CGM2_4LowerLimbs()
    model.configure()

    model.addAnthropoInputParameters(required_mp,optional=optional_mp)

    # --store calibration parameters--
    model.setStaticFilename(calibrateFilenameLabelled)
    model.setCalibrationProperty("leftFlatFoot",leftFlatFoot)
    model.setCalibrationProperty("rightFlatFoot",rightFlatFoot)
    model.setCalibrationProperty("markerDiameter",markerDiameter)


    # ---check marker set used----
    smc = cgm.CGM.checkCGM1_StaticMarkerConfig(acqStatic)


    # --------------------------STATIC CALBRATION--------------------------
    scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

    # ---initial calibration filter----
    # use if all optional mp are zero
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                        leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                                        markerDiameter=markerDiameter,
                                        ).compute()

    # ---- Decorators -----
    decorators.applyDecorators_CGM(smc, model,acqStatic,optional_mp,markerDiameter)
    decorators.applyHJCDecorators(model,hjcMethod)

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
        #                        ---OPENSIM IK---

        # --- opensim calibration Filter ---
        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
        markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-markerset.xml" # markerset
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure,
                                                str(DATA_PATH))
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build()


        # --- opensim Fitting Filter ---
        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-ikSetUp_template.xml" # ik tool file

        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model) # procedure
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

        cgmFittingProcedure.updateMarkerWeight("LTHAP",settings["Fitting"]["Weight"]["LTHAP"])
        cgmFittingProcedure.updateMarkerWeight("LTHAD",settings["Fitting"]["Weight"]["LTHAD"])
        cgmFittingProcedure.updateMarkerWeight("LTIAP",settings["Fitting"]["Weight"]["LTIAP"])
        cgmFittingProcedure.updateMarkerWeight("LTIAD",settings["Fitting"]["Weight"]["LTIAD"])
        cgmFittingProcedure.updateMarkerWeight("RTHAP",settings["Fitting"]["Weight"]["RTHAP"])
        cgmFittingProcedure.updateMarkerWeight("RTHAD",settings["Fitting"]["Weight"]["RTHAD"])
        cgmFittingProcedure.updateMarkerWeight("RTIAP",settings["Fitting"]["Weight"]["RTIAP"])
        cgmFittingProcedure.updateMarkerWeight("RTIAD",settings["Fitting"]["Weight"]["RTIAD"])

        cgmFittingProcedure.updateMarkerWeight("LSMH",settings["Fitting"]["Weight"]["LSMH"])
        cgmFittingProcedure.updateMarkerWeight("LFMH",settings["Fitting"]["Weight"]["LFMH"])
        cgmFittingProcedure.updateMarkerWeight("LVMH",settings["Fitting"]["Weight"]["LVMH"])

        cgmFittingProcedure.updateMarkerWeight("RSMH",settings["Fitting"]["Weight"]["RSMH"])
        cgmFittingProcedure.updateMarkerWeight("RFMH",settings["Fitting"]["Weight"]["RFMH"])
        cgmFittingProcedure.updateMarkerWeight("RVMH",settings["Fitting"]["Weight"]["RVMH"])

#            cgmFittingProcedure.updateMarkerWeight("LTHL",settings["Fitting"]["Weight"]["LTHL"])
#            cgmFittingProcedure.updateMarkerWeight("LTHLD",settings["Fitting"]["Weight"]["LTHLD"])
#            cgmFittingProcedure.updateMarkerWeight("LPAT",settings["Fitting"]["Weight"]["LPAT"])
#            cgmFittingProcedure.updateMarkerWeight("LTIBL",settings["Fitting"]["Weight"]["LTIBL"])
#            cgmFittingProcedure.updateMarkerWeight("RTHL",settings["Fitting"]["Weight"]["RTHL"])
#            cgmFittingProcedure.updateMarkerWeight("RTHLD",settings["Fitting"]["Weight"]["RTHLD"])
#            cgmFittingProcedure.updateMarkerWeight("RPAT",settings["Fitting"]["Weight"]["RPAT"])
#            cgmFittingProcedure.updateMarkerWeight("RTIBL",settings["Fitting"]["Weight"]["RTIBL"])


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
    validFrames,vff,vlf = btkTools.findValidFrames(acqGait,cgm2.CGM2_4LowerLimbs.TRACKING_MARKERS)



    # --- initial motion Filter ---
    scp=modelFilters.StaticCalibrationProcedure(model)
    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
    modMotion.compute()

    if ik_flag:
        #                        ---OPENSIM IK---

        # --- opensim calibration Filter ---
        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
        markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-markerset.xml" # markerset
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure,
                                                str(DATA_PATH))
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build()


        # --- opensim Fitting Filter ---
        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-ikSetUp_template.xml" # ik tl file

        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model) # procedure
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


        cgmFittingProcedure.updateMarkerWeight("LTHAP",settings["Fitting"]["Weight"]["LTHAP"])
        cgmFittingProcedure.updateMarkerWeight("LTHAD",settings["Fitting"]["Weight"]["LTHAD"])
        cgmFittingProcedure.updateMarkerWeight("LTIAP",settings["Fitting"]["Weight"]["LTIAP"])
        cgmFittingProcedure.updateMarkerWeight("LTIAD",settings["Fitting"]["Weight"]["LTIAD"])
        cgmFittingProcedure.updateMarkerWeight("RTHAP",settings["Fitting"]["Weight"]["RTHAP"])
        cgmFittingProcedure.updateMarkerWeight("RTHAD",settings["Fitting"]["Weight"]["RTHAD"])
        cgmFittingProcedure.updateMarkerWeight("RTIAP",settings["Fitting"]["Weight"]["RTIAP"])
        cgmFittingProcedure.updateMarkerWeight("RTIAD",settings["Fitting"]["Weight"]["RTIAD"])

        cgmFittingProcedure.updateMarkerWeight("LSMH",settings["Fitting"]["Weight"]["LSMH"])
        cgmFittingProcedure.updateMarkerWeight("LFMH",settings["Fitting"]["Weight"]["LFMH"])
        cgmFittingProcedure.updateMarkerWeight("LVMH",settings["Fitting"]["Weight"]["LVMH"])

        cgmFittingProcedure.updateMarkerWeight("RSMH",settings["Fitting"]["Weight"]["RSMH"])
        cgmFittingProcedure.updateMarkerWeight("RFMH",settings["Fitting"]["Weight"]["RFMH"])
        cgmFittingProcedure.updateMarkerWeight("RVMH",settings["Fitting"]["Weight"]["RVMH"])


#       cgmFittingProcedure.updateMarkerWeight("LTHL",settings["Fitting"]["Weight"]["LTHL"])
#       cgmFittingProcedure.updateMarkerWeight("LTHLD",settings["Fitting"]["Weight"]["LTHLD"])
#       cgmFittingProcedure.updateMarkerWeight("LPAT",settings["Fitting"]["Weight"]["LPAT"])
#       cgmFittingProcedure.updateMarkerWeight("LTIBL",settings["Fitting"]["Weight"]["LTIBL"])
#       cgmFittingProcedure.updateMarkerWeight("RTHL",settings["Fitting"]["Weight"]["RTHL"])
#       cgmFittingProcedure.updateMarkerWeight("RTHLD",settings["Fitting"]["Weight"]["RTHLD"])
#       cgmFittingProcedure.updateMarkerWeight("RPAT",settings["Fitting"]["Weight"]["RPAT"])
#       cgmFittingProcedure.updateMarkerWeight("RTIBL",settings["Fitting"]["Weight"]["RTIBL"])


        osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          scalingOsim,
                                                          cgmFittingProcedure,
                                                          str(DATA_PATH) )

        logging.info("-------INVERSE KINEMATICS IN PROGRESS----------")
        acqIK = osrf.run(acqGait,str(DATA_PATH + reconstructFilenameLabelled ))
        logging.info("-------INVERSE KINEMATICS DONE-----------------")



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
