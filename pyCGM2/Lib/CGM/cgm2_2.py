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
from pyCGM2.Model.CGM2 import decorators
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Model.Opensim import opensimFilters
from pyCGM2.Processing import progressionFrame
from pyCGM2.Signal import signal_processing



def calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
              required_mp,optional_mp,
              ik_flag,leftFlatFoot,rightFlatFoot,headFlat,
              markerDiameter,hjcMethod,
              pointSuffix,**kwargs):
    """
    Calibration of the CGM2.2

    :param DATA_PATH [str]: path to your data
    :param calibrateFilenameLabelled [str]: c3d file
    :param translators [dict]:  translators to apply
    :param required_mp [dict]: required anthropometric data
    :param optional_mp [dict]: optional anthropometric data (ex: LThighOffset,...)
    :param ik_flag [bool]: enable the inverse kinematic solver
    :param leftFlatFoot [bool]: enable of the flat foot option for the left foot
    :param rightFlatFoot [bool]: enable of the flat foot option for the right foot
    :param headFlat [bool]: enable of the head flat  option
    :param markerDiameter [double]: marker diameter (mm)
    :param hjcMethod [str or list of 3 float]: method for locating the hip joint centre
    :param pointSuffix [str]: suffix to add to model outputs

    """
    # --------------------ACQUISITION------------------------------

    # ---btk acquisition---

    if "forceBtkAcq" in kwargs.keys():
        acqStatic = kwargs["forceBtkAcq"]
    else:
        acqStatic = btkTools.smartReader((DATA_PATH+calibrateFilenameLabelled))


    btkTools.checkMultipleSubject(acqStatic)
    if btkTools.isPointExist(acqStatic,"SACR"):
        translators["LPSI"] = "SACR"
        translators["RPSI"] = "SACR"
        logging.info("[pyCGM2] Sacrum marker detected")

    acqStatic =  btkTools.applyTranslators(acqStatic,translators)

    # ---check marker set used----
    dcm= cgm.CGM.detectCalibrationMethods(acqStatic)

    # ---definition---
    model=cgm2.CGM2_2()
    model.configure(acq=acqStatic,detectedCalibrationMethods=dcm)

    model.addAnthropoInputParameters(required_mp,optional=optional_mp)
    # --store calibration parameters--
    model.setStaticFilename(calibrateFilenameLabelled)
    model.setCalibrationProperty("leftFlatFoot",leftFlatFoot)
    model.setCalibrationProperty("rightFlatFoot",rightFlatFoot)
    model.setCalibrationProperty("headFlat",headFlat)
    model.setCalibrationProperty("markerDiameter",markerDiameter)



    # --------------------------STATIC CALBRATION--------------------------
    scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

    # ---initial calibration filter----
    # use if all optional mp are zero
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                        leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                                        headFlat= headFlat,
                                        markerDiameter=markerDiameter,
                                        ).compute()

    # ---- Decorators -----
    decorators.applyBasicDecorators(dcm, model,acqStatic,optional_mp,markerDiameter)
    decorators.applyHJCDecorators(model,hjcMethod)

    # ----Final Calibration filter if model previously decorated -----
    if model.decoratedModel:
        # initial static filter
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                           headFlat= headFlat,
                           markerDiameter=markerDiameter).compute()

    # ----------------------CGM MODELLING----------------------------------
    # ----motion filter----
    modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,enums.motionMethod.Determinist,
                                              markerDiameter=markerDiameter)

    modMotion.compute()

    if "noKinematicsCalculation" in kwargs.keys() and kwargs["noKinematicsCalculation"]:
        logging.warning("[pyCGM2] No Kinematic calculation done for the static file")
        return model, acqStatic
    else:
        if model.getBodyPart() == enums.BodyPart.UpperLimb:
            ik_flag = False
            logging.warning("[pyCGM2] Fitting only applied for the upper limb")

        if ik_flag:
            #                        ---OPENSIM IK---

            # --- opensim calibration Filter ---
            osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
            markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm1\\cgm1-markerset.xml" # markerset
            cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

            oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                    model,
                                                    cgmCalibrationprocedure,
                                                    (DATA_PATH))
            oscf.addMarkerSet(markersetFile)
            scalingOsim = oscf.build()


            # --- opensim Fitting Filter ---
            iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm1\\cgm1-ikSetUp_template.xml" # ik tool file

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


            osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                              scalingOsim,
                                                              cgmFittingProcedure,
                                                              (DATA_PATH) )
            acqStaticIK = osrf.run(acqStatic,(DATA_PATH + calibrateFilenameLabelled ))



            # --- final pyCGM2 model motion Filter ---
            # use fitted markers
            modMotionFitted=modelFilters.ModelMotionFilter(scp,acqStaticIK,model,enums.motionMethod.Determinist)

            modMotionFitted.compute()


        # eventual static acquisition to consider for joint kinematics
        finalAcqStatic = acqStaticIK if ik_flag else acqStatic

        if "displayCoordinateSystem" in kwargs.keys() and kwargs["displayCoordinateSystem"]:
            csp = modelFilters.ModelCoordinateSystemProcedure(model)
            csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,finalAcqStatic)
            csdf.setStatic(False)
            csdf.display()

        #---- Joint kinematics----
        # relative angles
        modelFilters.ModelJCSFilter(model,finalAcqStatic).compute(description="vectoriel", pointLabelSuffix=pointSuffix)


        # detection of traveling axis + absolute angle
        if model.m_bodypart != enums.BodyPart.UpperLimb:
            pfp = progressionFrame.PelvisProgressionFrameProcedure()
        else:
            pfp = progressionFrame.ThoraxProgressionFrameProcedure()

        pff = progressionFrame.ProgressionFrameFilter(finalAcqStatic,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]


        if model.m_bodypart != enums.BodyPart.UpperLimb:
                modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqStatic,
                                                       segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                                        angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                        eulerSequences=["TOR","TOR", "ROT"],
                                                        globalFrameOrientation = globalFrame,
                                                        forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

        if model.m_bodypart == enums.BodyPart.LowerLimbTrunk:
                modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqStatic,
                                              segmentLabels=["Thorax"],
                                              angleLabels=["Thorax"],
                                              eulerSequences=["YXZ"],
                                              globalFrameOrientation = globalFrame,
                                              forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

        if model.m_bodypart == enums.BodyPart.UpperLimb or model.m_bodypart == enums.BodyPart.FullBody:

                modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqStatic,
                                              segmentLabels=["Thorax","Head"],
                                              angleLabels=["Thorax", "Head"],
                                              eulerSequences=["YXZ","TOR"],
                                              globalFrameOrientation = globalFrame,
                                              forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)
        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        if  model.m_bodypart == enums.BodyPart.FullBody:
            modelFilters.CentreOfMassFilter(model,finalAcqStatic).compute(pointLabelSuffix=pointSuffix)



        return model, finalAcqStatic


def fitting(model,DATA_PATH, reconstructFilenameLabelled,
    translators,settings,
    markerDiameter,
    pointSuffix,
    mfpa,
    momentProjection,**kwargs):

    """
    Fitting of the CGM2.2

    :param model [str]: pyCGM2 model previously calibrated
    :param DATA_PATH [str]: path to your data
    :param reconstructFilenameLabelled [string list]: c3d files
    :param translators [dict]:  translators to apply
    :param mfpa [str]: manual force plate assignement
    :param markerDiameter [double]: marker diameter (mm)
    :param pointSuffix [str]: suffix to add to model outputs
    :param momentProjection [str]: Coordinate system in which joint moment is expressed
    """


    # --------------------------ACQUISITION ------------------------------------

    # --- btk acquisition ----
    if "forceBtkAcq" in kwargs.keys():
        acqGait = kwargs["forceBtkAcq"]
    else:
        acqGait = btkTools.smartReader((DATA_PATH + reconstructFilenameLabelled))

    btkTools.checkMultipleSubject(acqGait)
    if btkTools.isPointExist(acqGait,"SACR"):
        translators["LPSI"] = "SACR"
        translators["RPSI"] = "SACR"
        logging.info("[pyCGM2] Sacrum marker detected")

    acqGait =  btkTools.applyTranslators(acqGait,translators)
    trackingMarkers = model.getTrackingMarkers(acqGait)
    validFrames,vff,vlf = btkTools.findValidFrames(acqGait,trackingMarkers)

    # filtering
    # -----------------------
    if "fc_lowPass_marker" in kwargs.keys() and kwargs["fc_lowPass_marker"]!=0 :
        fc = kwargs["fc_lowPass_marker"]
        order = 4
        if "order_lowPass_marker" in kwargs.keys():
            order = kwargs["order_lowPass_marker"]
        signal_processing.markerFiltering(acqGait,order=order, fc =fc)

    if "fc_lowPass_forcePlate" in kwargs.keys() and kwargs["fc_lowPass_forcePlate"]!=0 :
        fc = kwargs["fc_lowPass_forcePlate"]
        order = 4
        if "order_lowPass_forcePlate" in kwargs.keys():
            order = kwargs["order_lowPass_forcePlate"]
        signal_processing.forcePlateFiltering(acqGait,order=order, fc =fc)

    # --- initial motion Filter ---
    scp=modelFilters.StaticCalibrationProcedure(model)
    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
    modMotion.compute()

    ik_flag =  True
    if model.getBodyPart() == enums.BodyPart.UpperLimb:
        ik_flag = False
        logging.warning("[pyCGM2] Fitting only applied for the upper limb")

    if ik_flag:
        #                        ---OPENSIM IK---

        # --- opensim calibration Filter ---
        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
        markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm1\\cgm1-markerset.xml" # markerset
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure,
                                                (DATA_PATH))
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build()


        # --- opensim Fitting Filter ---
        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm1\\cgm1-ikSetUp_template.xml" # ik tool file

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


        osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          scalingOsim,
                                                          cgmFittingProcedure,
                                                          (DATA_PATH) )

        logging.info("-------INVERSE KINEMATICS IN PROGRESS----------")
        acqIK = osrf.run(acqGait,(DATA_PATH + reconstructFilenameLabelled ))
        logging.info("-------INVERSE KINEMATICS DONE-----------------")

    # eventual gait acquisition to consider for joint kinematics
    finalAcqGait = acqIK if ik_flag else acqGait

    # --- final pyCGM2 model motion Filter ---
    # use fitted markers
    modMotionFitted=modelFilters.ModelMotionFilter(scp,finalAcqGait,model,enums.motionMethod.Determinist ,
                                              markerDiameter=markerDiameter)

    modMotionFitted.compute()

    if "displayCoordinateSystem" in kwargs.keys() and kwargs["displayCoordinateSystem"]:
        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,finalAcqGait)
        csdf.setStatic(False)
        csdf.display()


    #---- Joint kinematics----
    # relative angles
    modelFilters.ModelJCSFilter(model,finalAcqGait).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

    # detection of traveling axis + absolute angle
    if model.m_bodypart != enums.BodyPart.UpperLimb:
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
    else:
        pfp = progressionFrame.ThoraxProgressionFrameProcedure()

    pff = progressionFrame.ProgressionFrameFilter(finalAcqGait,pfp)
    pff.compute()
    globalFrame = pff.outputs["globalFrame"]
    forwardProgression = pff.outputs["forwardProgression"]

    if model.m_bodypart != enums.BodyPart.UpperLimb:
            modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqGait,
                                                   segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                                    angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                    eulerSequences=["TOR","TOR", "ROT"],
                                                    globalFrameOrientation = globalFrame,
                                                    forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

    if model.m_bodypart == enums.BodyPart.LowerLimbTrunk:
            modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqGait,
                                          segmentLabels=["Thorax"],
                                          angleLabels=["Thorax"],
                                          eulerSequences=["YXZ"],
                                          globalFrameOrientation = globalFrame,
                                          forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

    if model.m_bodypart == enums.BodyPart.UpperLimb or model.m_bodypart == enums.BodyPart.FullBody:

            modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqGait,
                                          segmentLabels=["Thorax","Head"],
                                          angleLabels=["Thorax", "Head"],
                                          eulerSequences=["YXZ","TOR"],
                                          globalFrameOrientation = globalFrame,
                                          forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)


    #---- Body segment parameters----
    bspModel = bodySegmentParameters.Bsp(model)
    bspModel.compute()

    if  model.m_bodypart == enums.BodyPart.FullBody:
        modelFilters.CentreOfMassFilter(model,finalAcqGait).compute(pointLabelSuffix=pointSuffix)

    # Inverse dynamics
    if model.m_bodypart != enums.BodyPart.UpperLimb:
        # --- force plate handling----
        # find foot  in contact
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(finalAcqGait,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(finalAcqGait,mappedForcePlate)
        logging.warning("Manual Force plate assignment : %s" %mappedForcePlate)


        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,finalAcqGait,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix=pointSuffix)

        #---- Joint kinetics----
        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             finalAcqGait,
                             procedure = idp,
                             projection = momentProjection,
                             globalFrameOrientation = globalFrame,
                             forwardProgression = forwardProgression
                             ).compute(pointLabelSuffix=pointSuffix)


        #---- Joint energetics----
        modelFilters.JointPowerFilter(model,finalAcqGait).compute(pointLabelSuffix=pointSuffix)


    #---- zero unvalid frames ---
    btkTools.applyValidFramesOnOutput(finalAcqGait,validFrames)


    return finalAcqGait
