# -*- coding: utf-8 -*-
import os
import logging
import matplotlib.pyplot as plt
import json
import pdb
import cPickle
import json
from shutil import copyfile
from collections import OrderedDict
import argparse

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)



# pyCGM2 libraries
from pyCGM2.Tools import btkTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model.CGM2 import cgm,cgm2, modelFilters, modelDecorator
from pyCGM2.Utils import fileManagement
from pyCGM2.Model.Opensim import opensimFilters

if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM2.2 Calibration')
    parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('--ik', action='store_true', help='inverse kinematic',default=True)
    args = parser.parse_args()

    # --------------------GLOBAL SETTINGS ------------------------------

    # global setting ( in user/AppData)
    inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_2-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)


    # --------------------SESSION  SETTINGS ------------------------------
    DATA_PATH =os.getcwd()+"\\"
    infoSettings = json.loads(open('pyCGM2.info').read(),object_pairs_hook=OrderedDict)

    # --------------------CONFIGURATION ------------------------------

    hjcMethod = inputs["Calibration"]["HJC regression"]

    # ---- configuration parameters ----
    if args.leftFlatFoot is not None:
        flag_leftFlatFoot = bool(args.leftFlatFoot)
        logging.warning("Left flat foot forces : %s"%(str(bool(args.leftFlatFoot))))
    else:
        flag_leftFlatFoot = bool(inputs["Calibration"]["Left flat foot"])


    if args.rightFlatFoot is not None:
        flag_rightFlatFoot = bool(args.rightFlatFoot)
        logging.warning("Right flat foot forces : %s"%(str(bool(args.rightFlatFoot))))
    else:
        flag_rightFlatFoot =  bool(inputs["Calibration"]["Right flat foot"])


    if args.markerDiameter is not None:
        markerDiameter = float(args.markerDiameter)
        logging.warning("marker diameter forced : %s", str(float(args.markerDiameter)))
    else:
        markerDiameter = float(inputs["Global"]["Marker diameter"])


    if args.check:
        pointSuffix="cgm2.2"
    else:
        pointSuffix = inputs["Global"]["Point suffix"]

    

    # --------------------------TRANSLATORS ------------------------------------

    #  translators management
    translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM1-1.translators")
    if not translators:
       translators = inputs["Translators"]


    # --------------------------SUBJECT ------------------------------------

    required_mp={
    'Bodymass'   : infoSettings["MP"]["Required"]["Bodymass"],
    'LeftLegLength' :infoSettings["MP"]["Required"]["LeftLegLength"],
    'RightLegLength' : infoSettings["MP"]["Required"][ "RightLegLength"],
    'LeftKneeWidth' : infoSettings["MP"]["Required"][ "LeftKneeWidth"],
    'RightKneeWidth' : infoSettings["MP"]["Required"][ "RightKneeWidth"],
    'LeftAnkleWidth' : infoSettings["MP"]["Required"][ "LeftAnkleWidth"],
    'RightAnkleWidth' : infoSettings["MP"]["Required"][ "RightAnkleWidth"],
    'LeftSoleDelta' : infoSettings["MP"]["Required"][ "LeftSoleDelta"],
    'RightSoleDelta' : infoSettings["MP"]["Required"]["RightSoleDelta"]
    }

    optional_mp={
    'InterAsisDistance'   : infoSettings["MP"]["Optional"][ "InterAsisDistance"],#0,
    'LeftAsisTrocanterDistance' : infoSettings["MP"]["Optional"][ "LeftAsisTrocanterDistance"],#0,
    'LeftTibialTorsion' : infoSettings["MP"]["Optional"][ "LeftTibialTorsion"],#0 ,
    'LeftThighRotation' : infoSettings["MP"]["Optional"][ "LeftThighRotation"],#0,
    'LeftShankRotation' : infoSettings["MP"]["Optional"][ "LeftShankRotation"],#0,
    'RightAsisTrocanterDistance' : infoSettings["MP"]["Optional"][ "RightAsisTrocanterDistance"],#0,
    'RightTibialTorsion' : infoSettings["MP"]["Optional"][ "RightTibialTorsion"],#0 ,
    'RightThighRotation' : infoSettings["MP"]["Optional"][ "RightThighRotation"],#0,
    'RightShankRotation' : infoSettings["MP"]["Optional"][ "RightShankRotation"],#0,
        }


    # --------------------------ACQUISITION--------------------------------------

    calibrateFilenameLabelled = infoSettings["Modelling"]["Trials"]["Static"]

    logging.info( "data Path: "+ DATA_PATH )
    logging.info( "calibration file: "+ calibrateFilenameLabelled)

    # ---btk acquisition---
    acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))
    btkTools.checkMultipleSubject(acqStatic)

    acqStatic =  btkTools.applyTranslators(acqStatic,translators)


    # ---definition---
    model=cgm2.CGM2_2LowerLimbs()
    model.configure()
    model.addAnthropoInputParameters(required_mp,optional=optional_mp)


    # ---check marker set used----
    staticMarkerConfiguration= cgm.CGM.checkCGM1_StaticMarkerConfig(acqStatic)


    # --------------------------STATIC CALBRATION--------------------------
    scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

    # ---initial calibration filter----
    # use if all optional mp are zero
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                        leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                                        markerDiameter=markerDiameter,
                                        ).compute()
    # ---- Decorators -----
    # Goal = modified calibration according the identified marker set or if offsets manually set

    # hip joint centres ---
    useLeftHJCnodeLabel = "LHJC_cgm1"
    useRightHJCnodeLabel = "RHJC_cgm1"
    if hjcMethod == "Hara":
        modelDecorator.HipJointCenterDecorator(model).hara()
        useLeftHJCnodeLabel = "LHJC_Hara"
        useRightHJCnodeLabel = "RHJC_Hara"


    # knee - ankle centres ---

    # initialisation of node label
    useLeftKJCnodeLabel = "LKJC_chord"
    useLeftAJCnodeLabel = "LAJC_chord"
    useRightKJCnodeLabel = "RKJC_chord"
    useRightAJCnodeLabel = "RAJC_chord"

    # case 1 : NO kad, NO medial ankle BUT thighRotation different from zero ( mean manual modification or new calibration from a previous one )
    if not staticMarkerConfiguration["leftKadFlag"]  and not staticMarkerConfiguration["leftMedialAnkleFlag"] and not staticMarkerConfiguration["leftMedialKneeFlag"] and optional_mp["LeftThighRotation"] !=0:
        logging.warning("CASE FOUND ===> Left Side - CGM1 - Origine - manual offsets")
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",optional_mp["LeftThighRotation"],markerDiameter,optional_mp["LeftTibialTorsion"],optional_mp["LeftShankRotation"])
        useLeftKJCnodeLabel = "LKJC_mo"
        useLeftAJCnodeLabel = "LAJC_mo"
    if not staticMarkerConfiguration["rightKadFlag"]  and not staticMarkerConfiguration["rightMedialAnkleFlag"] and not staticMarkerConfiguration["rightMedialKneeFlag"] and optional_mp["RightThighRotation"] !=0:
        logging.warning("CASE FOUND ===> Right Side - CGM1 - Origine - manual offsets")
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",optional_mp["RightThighRotation"],markerDiameter,optional_mp["RightTibialTorsion"],optional_mp["RightShankRotation"])
        useRightKJCnodeLabel = "RKJC_mo"
        useRightAJCnodeLabel = "RAJC_mo"

    # case 2 : kad FOUND and NO medial Ankle
    if staticMarkerConfiguration["leftKadFlag"]:
        logging.warning("CASE FOUND ===> Left Side - CGM1 - KAD variant")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")
        useLeftKJCnodeLabel = "LKJC_kad"
        useLeftAJCnodeLabel = "LAJC_kad"
    if staticMarkerConfiguration["rightKadFlag"]:
        logging.warning("CASE FOUND ===> Right Side - CGM1 - KAD variant")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")
        useRightKJCnodeLabel = "RKJC_kad"
        useRightAJCnodeLabel = "RAJC_kad"


    # case 3 : medial knee FOUND
    # notice: cgm1Behaviour is enable mean effect on AJC location
    if staticMarkerConfiguration["leftMedialKneeFlag"]:
        logging.warning("CASE FOUND ===> Left Side - CGM1 - medial knee ")
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="left",cgm1Behaviour=True)
        useLeftKJCnodeLabel = "LKJC_mid"
        useLeftAJCnodeLabel = "LAJC_midKnee"
    if staticMarkerConfiguration["rightMedialKneeFlag"]:
        logging.warning("CASE FOUND ===> Right Side - CGM1 - medial knee ")
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="right",cgm1Behaviour=True)
        useRightKJCnodeLabel = "RKJC_mid"
        useRightAJCnodeLabel = "RAJC_midKnee"


    # case 4 : medial ankle FOUND
    if staticMarkerConfiguration["leftMedialAnkleFlag"]:
        logging.warning("CASE FOUND ===> Left Side - CGM1 - medial ankle ")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
        useLeftAJCnodeLabel = "LAJC_mid"
    if staticMarkerConfiguration["rightMedialAnkleFlag"]:
        logging.warning("CASE FOUND ===> Right Side - CGM1 - medial ankle ")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")
        useRightAJCnodeLabel = "RAJC_mid"

    properties_initialCalibration=dict()
    properties_initialCalibration["LHJC_node"] = useLeftHJCnodeLabel
    properties_initialCalibration["RHJC_node"] = useRightHJCnodeLabel
    properties_initialCalibration["LKJC_node"] = useLeftKJCnodeLabel
    properties_initialCalibration["RKJC_node"] = useRightKJCnodeLabel
    properties_initialCalibration["LAJC_node"] = useLeftAJCnodeLabel
    properties_initialCalibration["RAJC_node"] = useRightAJCnodeLabel
    properties_initialCalibration["rightFlatFoot"] = useRightAJCnodeLabel
    properties_initialCalibration["leftFlatFoot"] = flag_rightFlatFoot
    properties_initialCalibration["markerDiameter"] = markerDiameter


    # ----Final Calibration filter if model previously decorated -----
    if model.decoratedModel:
        # initial static filter
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           useLeftKJCnode=useLeftKJCnodeLabel, useLeftAJCnode=useLeftAJCnodeLabel,
                           useRightKJCnode=useRightKJCnodeLabel, useRightAJCnode=useRightAJCnodeLabel,
                           leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                           markerDiameter=markerDiameter).compute()

    # set initial calibration as model property
    model.m_properties["CalibrationParameters0"] = properties_initialCalibration


    # ----------------------CGM MODELLING----------------------------------
    if args.ik:
        #                        ---OPENSIM IK---

        # --- opensim calibration Filter ---
        osimfile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
        markersetFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm1\\cgm1-markerset.xml" # markerset
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure)
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build()


        # --- opensim Fitting Filter ---
        iksetupFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm1\\cgm1-ikSetUp_template.xml" # ik tool file

        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model) # procedure
        cgmFittingProcedure.updateMarkerWeight("LASI",inputs["Fitting"]["Weight"]["LASI"])
        cgmFittingProcedure.updateMarkerWeight("RASI",inputs["Fitting"]["Weight"]["RASI"])
        cgmFittingProcedure.updateMarkerWeight("LPSI",inputs["Fitting"]["Weight"]["LPSI"])
        cgmFittingProcedure.updateMarkerWeight("RPSI",inputs["Fitting"]["Weight"]["RPSI"])
        cgmFittingProcedure.updateMarkerWeight("RTHI",inputs["Fitting"]["Weight"]["RTHI"])
        cgmFittingProcedure.updateMarkerWeight("RKNE",inputs["Fitting"]["Weight"]["RKNE"])
        cgmFittingProcedure.updateMarkerWeight("RTIB",inputs["Fitting"]["Weight"]["RTIB"])
        cgmFittingProcedure.updateMarkerWeight("RANK",inputs["Fitting"]["Weight"]["RANK"])
        cgmFittingProcedure.updateMarkerWeight("RHEE",inputs["Fitting"]["Weight"]["RHEE"])
        cgmFittingProcedure.updateMarkerWeight("RTOE",inputs["Fitting"]["Weight"]["RTOE"])
        cgmFittingProcedure.updateMarkerWeight("LTHI",inputs["Fitting"]["Weight"]["LTHI"])
        cgmFittingProcedure.updateMarkerWeight("LKNE",inputs["Fitting"]["Weight"]["LKNE"])
        cgmFittingProcedure.updateMarkerWeight("LTIB",inputs["Fitting"]["Weight"]["LTIB"])
        cgmFittingProcedure.updateMarkerWeight("LANK",inputs["Fitting"]["Weight"]["LANK"])
        cgmFittingProcedure.updateMarkerWeight("LHEE",inputs["Fitting"]["Weight"]["LHEE"])
        cgmFittingProcedure.updateMarkerWeight("LTOE",inputs["Fitting"]["Weight"]["LTOE"])


        osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          scalingOsim,
                                                          cgmFittingProcedure,
                                                          str(DATA_PATH) )
        acqStaticIK = osrf.run(acqStatic,str(DATA_PATH + calibrateFilenameLabelled ))



        # --- final pyCGM2 model motion Filter ---
        # use fitted markers
        modMotionFitted=modelFilters.ModelMotionFilter(scp,acqStaticIK,model,pyCGM2Enums.motionMethod.Sodervisk)

        modMotionFitted.compute()


    # eventual static acquisition to consider for joint kinematics
    finalAcqStatic = acqStaticIK if args.ik else acqStatic

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


    # ----------------------SAVE-------------------------------------------


    # update optional mp and save a new info file
    infoSettings["MP"]["Optional"][ "InterAsisDistance"] = model.mp_computed["InterAsisDistance"]
    infoSettings["MP"]["Optional"][ "LeftAsisTrocanterDistance"] = model.mp_computed["LeftAsisTrocanterDistance"]
    infoSettings["MP"]["Optional"][ "LeftTibialTorsion"] = model.mp_computed["LeftTibialTorsionOffset"]
    infoSettings["MP"]["Optional"][ "LeftThighRotation"] = model.mp_computed["LeftThighRotationOffset"]
    infoSettings["MP"]["Optional"][ "LeftShankRotation"] = model.mp_computed["LeftShankRotationOffset"]
    infoSettings["MP"]["Optional"][ "RightAsisTrocanterDistance"] = model.mp_computed["RightAsisTrocanterDistance"]
    infoSettings["MP"]["Optional"][ "RightTibialTorsion"] = model.mp_computed["RightTibialTorsionOffset"]
    infoSettings["MP"]["Optional"][ "RightThighRotation"] = model.mp_computed["RightThighRotationOffset"]
    infoSettings["MP"]["Optional"][ "RightShankRotation"] = model.mp_computed["RightShankRotationOffset"]

    with open('pyCGM2.info', 'w') as outfile:
        json.dump(infoSettings, outfile,indent=4)


    # save pycgm2 -model
    if os.path.isfile(DATA_PATH + "pyCGM2.model"):
        logging.warning("previous model removed")
        os.remove(DATA_PATH + "pyCGM2.model")

    modelFile = open(DATA_PATH + "pyCGM2.model", "w")
    cPickle.dump(model, modelFile)
    modelFile.close()




    #  static file
    btkTools.smartWriter(finalAcqStatic, str(DATA_PATH+calibrateFilenameLabelled[:-4]+"-modelled.c3d"))
