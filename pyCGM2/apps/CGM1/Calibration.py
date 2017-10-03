# -*- coding: utf-8 -*-
import os
import logging
import matplotlib.pyplot as plt
import argparse

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.DEBUG)

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Eclipse import vskTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model import modelFilters, modelDecorator
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Utils import files,infoFile

if __name__ == "__main__":

    DEBUG = False
    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM1 Calibration')
    parser.add_argument('--infoFile', type=str, help='infoFile')
    parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix' )
    parser.add_argument('-fs','--fileSuffix', type=str, help='suffix of output file')
    parser.add_argument('--vsk', type=str, help='vicon skeleton filename')
    parser.add_argument('--resetMP', action='store_false', help='reset optional mass parameters')
    args = parser.parse_args()


    # --------------------GLOBAL SETTINGS ------------------------------

    # global setting ( in user/AppData)
    settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.settings")
    #json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM1-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)


    # --------------------SESSION  SETTINGS ------------------------------
    if DEBUG:
        DATA_PATH = "C:\\Users\\HLS501\\Google Drive\\Paper_for BJSM\\BJSM_trials\\FMS_Screening\\15KUFC01\\Session 2\\"
        infoFilename = "pyCGM2.info"
        info = files.openJson(DATA_PATH,infoFilename)
        args.vsk = "15KUFC01.vsk"
    else:
        DATA_PATH =os.getcwd()+"\\"
        infoFilename = "pyCGM2.info" if args.infoFile is None else  args.infoFile
        info = files.openJson(DATA_PATH,infoFilename)

    # --------------------CONFIGURATION ------------------------------

    # ---- configuration parameters ----
    if args.leftFlatFoot is not None:
        flag_leftFlatFoot = bool(args.leftFlatFoot)
        logging.warning("Left flat foot forces : %s"%(str(bool(args.leftFlatFoot))))
    else:
        flag_leftFlatFoot = bool(settings["Calibration"]["Left flat foot"])


    if args.rightFlatFoot is not None:
        flag_rightFlatFoot = bool(args.rightFlatFoot)
        logging.warning("Right flat foot forces : %s"%(str(bool(args.rightFlatFoot))))
    else:
        flag_rightFlatFoot =  bool(settings["Calibration"]["Right flat foot"])


    if args.markerDiameter is not None:
        markerDiameter = float(args.markerDiameter)
        logging.warning("marker diameter forced : %s", str(float(args.markerDiameter)))
    else:
        markerDiameter = float(settings["Global"]["Marker diameter"])


    if args.check:
        pointSuffix="cgm1.0"
    else:
        if args.pointSuffix is not None:
            pointSuffix = args.pointSuffix
        else:
            pointSuffix = settings["Global"]["Point suffix"]



    # --------------------------TRANSLATORS ------------------------------------

    #  translators management
    translators = files.manage_pycgm2Translators(DATA_PATH,"CGM1.translators")
    if not translators:
       translators = settings["Translators"]


    # --------------------------SUBJECT ------------------------------------
    if args.vsk is  None:
        required_mp,optional_mp = infoFile.getFromInfoSubjectMp(info, resetFlag=args.resetMP)
    else:
        vsk = vskTools.Vsk(str(DATA_PATH + args.vsk))
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=args.resetMP)

    # --------------------------ACQUISITION--------------------------------------

    calibrateFilenameLabelled = info["Modelling"]["Trials"]["Static"]

    logging.info( "data Path: "+ DATA_PATH )
    logging.info( "calibration file: "+ calibrateFilenameLabelled)

    # ---btk acquisition---
    acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))
    btkTools.checkMultipleSubject(acqStatic)

    acqStatic =  btkTools.applyTranslators(acqStatic,translators)


    # ---definition---
    model=cgm.CGM1LowerLimbs()
    model.configure()
    model.addAnthropoInputParameters(required_mp,optional=optional_mp)
    model.setStaticFilename(calibrateFilenameLabelled)

    # ---check marker set used----
    staticMarkerConfiguration= cgm.CGM.checkCGM1_StaticMarkerConfig(acqStatic)

    # --------------------------STATIC CALBRATION--------------------------
    scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

    # ---initial calibration filter----
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                        leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                                        markerDiameter=markerDiameter,
                                        ).compute()
    # ---- Decorators -----
    # Goal = modified calibration according the identified marker set or if offsets manually set

    # initialisation of node label and marker labels

    useLeftHJCnodeLabel = "LHJC_cgm1"
    useRightHJCnodeLabel = "RHJC_cgm1"

    useLeftKJCnodeLabel = "LKJC_chord"
    useLeftAJCnodeLabel = "LAJC_chord"
    useRightKJCnodeLabel = "RKJC_chord"
    useRightAJCnodeLabel = "RAJC_chord"

    useLeftKJCmarkerLabel = "LKJC"
    useLeftAJCmarkerLabel = "LAJC"
    useRightKJCmarkerLabel = "RKJC"
    useRightAJCmarkerLabel = "RAJC"


    # case 1 : NO kad, NO medial ankle BUT thighRotation different from zero ( mean manual modification or new calibration from a previous one )
    #  case not necessary - static PIG operation - dont consider any offsets
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

        useLeftKJCmarkerLabel = "LKJC_KAD"
        useLeftAJCmarkerLabel = "LAJC_KAD"

    if staticMarkerConfiguration["rightKadFlag"]:
        logging.warning("CASE FOUND ===> Right Side - CGM1 - KAD variant")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")
        useRightKJCnodeLabel = "RKJC_kad"
        useRightAJCnodeLabel = "RAJC_kad"

        useRightKJCmarkerLabel = "RKJC_KAD"
        useRightAJCmarkerLabel = "RAJC_KAD"


    # case 3 : both kad and medial ankle FOUND
    if staticMarkerConfiguration["leftKadFlag"]:
        if staticMarkerConfiguration["leftMedialAnkleFlag"]:
            logging.warning("CASE FOUND ===> Left Side - CGM1 - KAD + medial ankle ")
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
            useLeftAJCnodeLabel = "LAJC_mid"

            useLeftAJCmarkerLabel = "LAJC_MID"


    if staticMarkerConfiguration["rightKadFlag"]:
        if staticMarkerConfiguration["rightMedialAnkleFlag"]:
            logging.warning("CASE FOUND ===> Right Side - CGM1 - KAD + medial ankle ")
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")
            useRightAJCnodeLabel = "RAJC_mid"

            useRightAJCmarkerLabel = "RAJC_MID"

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
    # ----motion filter----
    # notice : viconCGM1compatible option duplicate error on Construction of the foot coordinate system

    modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,pyCGM2Enums.motionMethod.Determinist,
                                              markerDiameter=markerDiameter,
                                              viconCGM1compatible=False,
                                              pigStatic=True,
                                              useRightKJCmarker=useRightKJCmarkerLabel, useRightAJCmarker=useRightAJCmarkerLabel,
                                              useLeftKJCmarker=useLeftKJCmarkerLabel, useLeftAJCmarker=useLeftAJCmarkerLabel)
    modMotion.compute()


    #---- Joint kinematics----
    # relative angles
    modelFilters.ModelJCSFilter(model,acqStatic).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

    # detection of traveling axis
    longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqStatic,["LASI","RASI","RPSI","LPSI"])

    # absolute angles
    modelFilters.ModelAbsoluteAnglesFilter(model,acqStatic,
                                           segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                            angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                            eulerSequences=["TOR","TOR", "TOR"],
                                            globalFrameOrientation = globalFrame,
                                            forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)



    # ----------------------SAVE-------------------------------------------


    # update optional mp and save a new info file
    info["MP"]["Optional"][ "InterAsisDistance"] = model.mp_computed["InterAsisDistance"]
    info["MP"]["Optional"][ "LeftAsisTrocanterDistance"] = model.mp_computed["LeftAsisTrocanterDistance"]
    info["MP"]["Optional"][ "LeftTibialTorsion"] = model.mp_computed["LeftTibialTorsionOffset"]
    info["MP"]["Optional"][ "LeftThighRotation"] = model.mp_computed["LeftThighRotationOffset"]
    info["MP"]["Optional"][ "LeftShankRotation"] = model.mp_computed["LeftShankRotationOffset"]
    info["MP"]["Optional"][ "RightAsisTrocanterDistance"] = model.mp_computed["RightAsisTrocanterDistance"]
    info["MP"]["Optional"][ "RightTibialTorsion"] = model.mp_computed["RightTibialTorsionOffset"]
    info["MP"]["Optional"][ "RightThighRotation"] = model.mp_computed["RightThighRotationOffset"]
    info["MP"]["Optional"][ "RightShankRotation"] = model.mp_computed["RightShankRotationOffset"]

    files.saveJson(DATA_PATH, infoFilename, info)


    # save pycgm2 -model
    files.saveModel(model,DATA_PATH,None)


    # new static file
    if args.fileSuffix is not None:
        btkTools.smartWriter(acqStatic, str(DATA_PATH+calibrateFilenameLabelled[:-4]+"-modelled-"+args.fileSuffix+".c3d"))
    else:
        btkTools.smartWriter(acqStatic, str(DATA_PATH+calibrateFilenameLabelled[:-4]+"-modelled.c3d"))
