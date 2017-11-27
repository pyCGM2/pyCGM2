leftFlatFoot# -*- coding: utf-8 -*-
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
from pyCGM2.apps import cgmUtils

if __name__ == "__main__":

    DEBUG = False
    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM1.1 Calibration')
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
    settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM1_1-pyCGM2.settings")



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

    # --------------------------CONFIG ------------------------------------
    argsManager = cgmUtils.argsManager_cgm1(settings,args)
    leftFlatFoot = argsManager.getLeftFlatFoot()
    rightFlatFoot = argsManager.getRightFlatFoot()
    markerDiameter = argsManager.getMarkerDiameter()
    pointSuffix = argsManager.getPointSuffix("cgm1.1")



    # --------------------------TRANSLATORS ------------------------------------

    #  translators management
    translators = files.manage_pycgm2Translators(DATA_PATH,"CGM1-1.translators")
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
    model.setVersion("CGM1.1")
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
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                        leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                                        markerDiameter=markerDiameter,
                                        ).compute()
    # ---- Decorators -----
    cgmUtils.applyDecorators_CGM(smc, model,acqStatic,optional_mp,markerDiameter)

    # ----Final Calibration filter if model previously decorated -----
    if model.decoratedModel:
        # initial static filter
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                           markerDiameter=markerDiameter).compute()


    # ----------------------CGM MODELLING----------------------------------
    # ----motion filter----
    # notice : viconCGM1compatible option duplicate error on Construction of the foot coordinate system

    modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,enums.motionMethod.Determinist,
                                              markerDiameter=markerDiameter)
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


    info["MP"]["Optional"][ "LeftKneeFuncCalibrationOffset"] = model.mp_computed["LeftKneeFuncCalibrationOffset"]
    info["MP"]["Optional"][ "RightKneeFuncCalibrationOffset"] = model.mp_computed["RightKneeFuncCalibrationOffset"]


    files.saveJson(DATA_PATH, infoFilename, info)


    # save pycgm2 -model
    files.saveModel(model,DATA_PATH,None)


    # new static file
    if args.fileSuffix is not None:
        btkTools.smartWriter(acqStatic, str(DATA_PATH+calibrateFilenameLabelled[:-4]+"-modelled-"+args.fileSuffix+".c3d"))
    else:
        btkTools.smartWriter(acqStatic, str(DATA_PATH+calibrateFilenameLabelled[:-4]+"-modelled.c3d"))
