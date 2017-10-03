# -*- coding: utf-8 -*-
import sys
import pdb
import logging
import matplotlib.pyplot as plt
import argparse
import json
import os
import numpy as np

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Model import modelFilters, modelDecorator
from pyCGM2.Model.CGM2 import cgm, cgm2
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Utils import files,infoFile

def detectSide(acq,left_markerLabel,right_markerLabel):

    flag,vff,vlf = btkTools.findValidFrames(acq,[left_markerLabel,right_markerLabel])

    left = acq.GetPoint(left_markerLabel).GetValues()[vff:vlf,2]
    right = acq.GetPoint(right_markerLabel).GetValues()[vff:vlf,2]

    side = "Left" if np.max(left)>np.max(right) else "Right"

    return side


if __name__ == "__main__":

    plt.close("all")
    DEBUG = False

    parser = argparse.ArgumentParser(description='2Dof Knee Calibration')
    parser.add_argument('--infoFile', type=str, help='infoFile')
    args = parser.parse_args()

    # --------------------SESSION SETTINGS ------------------------------
    if DEBUG:
        DATA_PATH = "C:\\Users\\HLS501\\Google Drive\\Paper_for BJSM\\BJSM_trials\\FMS_Screening\\15KUFC01\\Session 2\\"
        infoFilename = "pyCGM2.info"
        info = files.openJson(DATA_PATH,infoFilename)

    else:
        DATA_PATH =os.getcwd()+"\\"
        infoFilename = "pyCGM2.info" if args.infoFile is None else  args.infoFile
        info = files.openJson(DATA_PATH,infoFilename)



    # --------------------CONFIGURATION ------------------------------
    # NA

    # --------------------pyCGM2 MODEL------------------------------

    model = files.loadModel(DATA_PATH,None)

    logging.info("loaded model : %s" %(model.version ))

    # --------------------GLOBAL SETTNGS ------------------------------

    if model.version == "CGM1.0":
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.settings")
    elif model.version == "CGM1.1":
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM1_1-pyCGM2.settings")
    elif model.version == "CGM2.1":
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_1-pyCGM2.settings")
    elif model.version == "CGM2.2":
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_2-pyCGM2.settings")
    elif model.version == "CGM2.2e":
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_2-Expert-pyCGM2.settings")
    elif model.version == "CGM2.3":
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_3-pyCGM2.settings")
    elif model.version == "CGM2.3e":
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_3-Expert-pyCGM2.settings")
    elif model.version == "CGM2.4":
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_4-pyCGM2.settings")
    elif model.version == "CGM2.4e":
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_4-Expert-pyCGM2.settings")
    else:
        raise Exception ("model version not found [contact admin]")

    # --------------------------TRANSLATORS ------------------------------------

    #  translators management
    if model.version in  ["CGM1.0"]:
        translators = files.manage_pycgm2Translators(DATA_PATH,"CGM1.translators")
    elif model.version in  ["CGM1.1","CGM2.1","CGM2.2","CGM2.2e"]:
        translators = files.manage_pycgm2Translators(DATA_PATH,"CGM1-1.translators")
    elif model.version in  ["CGM2.3","CGM2.3e"]:
        translators = files.manage_pycgm2Translators(DATA_PATH,"CGM2-3.translators")
    elif model.version in  ["CGM2.4","CGM2.4e"]:
        translators = files.manage_pycgm2Translators(DATA_PATH,"CGM2-4.translators")

    if not translators:
       translators = settings["Translators"]

    # --------------------------ACQ WITH TRANSLATORS --------------------------------------
    funcTrials = info["Modelling"]["KneeCalibrationTrials"]["Calibration2Dof"]

    for it in funcTrials:

        trial = it["Trial"]

        # --- btk acquisition ----
        print str(DATA_PATH + trial)
        acqFunc = btkTools.smartReader(str(DATA_PATH + trial))
        acqFunc =  btkTools.applyTranslators(acqFunc,translators)

        #---get frame range of interest---
        ff = acqFunc.GetFirstFrame()
        lf = acqFunc.GetLastFrame()

        initFrame = int(it["beginFrame"]) if it["beginFrame"] is not "" else ff
        endFrame = int(it["endFrame"]) if it["endFrame"] is not "" else lf

        iff=initFrame-ff
        ilf=endFrame-ff

        # motion
        if it["Side"]=="":
            side = detectSide(acqFunc,"LANK","RANK")
            logging.info("Detected motion side : %s" %(side) )
        else:
            side = it["Side"]

        if model.version in  ["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e"]:
            validFrames,vff,vlf = btkTools.findValidFrames(acqFunc,cgm.CGM1LowerLimbs.MARKERS)


        # --------------------------RESET OF THE STATIC File---------

        # load btkAcq from static file
        staticFilename = model.m_staticFilename
        acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilename))
        btkTools.checkMultipleSubject(acqStatic)
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)

        # initial calibration ( i.e calibration from Calibration operation)
        initialCalibration = model.m_properties["CalibrationParameters0"]
        flag_leftFlatFoot = initialCalibration["leftFlatFoot"]
        flag_rightFlatFoot = initialCalibration["rightFlatFoot"]
        markerDiameter = initialCalibration["markerDiameter"]

        if side == "Left":
            # remove other functional calibration
            model.mp_computed["LeftKneeFuncCalibrationOffset"] = 0

            # reinit node and func offset of the left side from initial calibration
            useLeftHJCnodeLabel = initialCalibration["LHJC_node"]
            useLeftKJCnodeLabel = initialCalibration["LKJC_node"]
            useLeftAJCnodeLabel = initialCalibration["LAJC_node"]
            model.mp_computed["LeftKnee2DofOffset"] = 0
            model.mp_computed["FinalFuncLeftThighRotationOffset"] =0

            # opposite side - keep node from former calibration
            if model.mp_computed["RightKnee2DofOffset"]:
                useRightHJCnodeLabel = model.m_properties["CalibrationParameters"]["RHJC_node"]
                useRightKJCnodeLabel = model.m_properties["CalibrationParameters"]["RKJC_node"]
                useRightAJCnodeLabel = model.m_properties["CalibrationParameters"]["RAJC_node"]

            else:
                useRightHJCnodeLabel = initialCalibration["RHJC_node"]
                useRightKJCnodeLabel = initialCalibration["RKJC_node"]
                useRightAJCnodeLabel = initialCalibration["RAJC_node"]

        if side == "Right":
            # remove other functional calibration
            model.mp_computed["RightKneeFuncCalibrationOffset"] = 0

            # reinit node and func offset of the right side from initial calibration
            useRightHJCnodeLabel = initialCalibration["RHJC_node"]
            useRightKJCnodeLabel = initialCalibration["RKJC_node"]
            useRightAJCnodeLabel = initialCalibration["RAJC_node"]
            model.mp_computed["RightKnee2DofOffset"] = 0
            model.mp_computed["FinalFuncRightThighRotationOffset"] =0

            # opposite side - keep node from former calibration
            if model.mp_computed["LeftKnee2DofOffset"]:
                useLeftHJCnodeLabel = model.m_properties["CalibrationParameters"]["LHJC_node"]
                useLeftKJCnodeLabel = model.m_properties["CalibrationParameters"]["LKJC_node"]
                useLeftAJCnodeLabel = model.m_properties["CalibrationParameters"]["LAJC_node"]

            else:
                useLeftHJCnodeLabel = initialCalibration["LHJC_node"]
                useLeftKJCnodeLabel = initialCalibration["LKJC_node"]
                useLeftAJCnodeLabel = initialCalibration["LAJC_node"]


        # ---- Reset Calibration
        logging.debug("Calibration2Dof --%s --- first Calibration"%(side) )
        logging.debug(" node (LHJC) => %s" %(useLeftHJCnodeLabel))
        logging.debug(" node (LKJC) => %s" %(useLeftKJCnodeLabel))
        logging.debug(" node (LAJC) => %s" %(useLeftAJCnodeLabel))
        logging.debug("-opposite side-" )
        logging.debug(" node (RHJC) => %s" %(useRightHJCnodeLabel))
        logging.debug(" node (RKJC) => %s" %(useRightKJCnodeLabel))
        logging.debug(" node (RAJC) => %s" %(useRightAJCnodeLabel))



        # no rotation of both thigh
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                               useLeftHJCnode=useLeftHJCnodeLabel, useRightHJCnode=useRightHJCnodeLabel,
                               useLeftKJCnode=useLeftKJCnodeLabel, useLeftAJCnode=useLeftAJCnodeLabel,
                               useRightKJCnode=useRightKJCnodeLabel, useRightAJCnode=useRightAJCnodeLabel,
                               leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                               markerDiameter=markerDiameter,
                               RotateLeftThighFlag = False,
                               RotateRightThighFlag = False).compute()




        if model.version in  ["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e"]:

            modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,pyCGM2Enums.motionMethod.Determinist,
                                                     markerDiameter=markerDiameter,
                                                     viconCGM1compatible=True)
            modMotion.compute()




        elif model.version in  ["CGM2.3","CGM2.3e","CGM2.3","CGM2.4e"]:
            if side == "Left":
                thigh_markers = model.getSegment("Left Thigh").m_tracking_markers
                shank_markers = model.getSegment("Left Shank").m_tracking_markers

            elif side == "Right":
                thigh_markers = model.getSegment("Right Thigh").m_tracking_markers
                shank_markers = model.getSegment("Right Shank").m_tracking_markers

            validFrames,vff,vlf = btkTools.findValidFrames(acqFunc,thigh_markers+shank_markers)

            proximalSegmentLabel=str(side+" Thigh")
            distalSegmentLabel=str(side+" Shank")

            # Motion
            modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,pyCGM2Enums.motionMethod.Sodervisk)
            modMotion.segmentalCompute([proximalSegmentLabel,distalSegmentLabel])


        # calibration decorators
        modelDecorator.KneeCalibrationDecorator(model).calibrate2dof(side,
                                                                       indexFirstFrame = iff,
                                                                       indexLastFrame = ilf)


        # --------------------------FINAL CALIBRATION OF THE STATIC File---------

        if side == "Left":
            useLeftHJCnodeLabel = model.m_properties["CalibrationParameters"]["LHJC_node"]
            useLeftKJCnodeLabel = model.m_properties["CalibrationParameters"]["LKJC_node"]
            useLeftAJCnodeLabel = model.m_properties["CalibrationParameters"]["LAJC_node"]
            useRotateLeftThighFlag = True
            useRotateRightThighFlag = False

        elif side == "Right":
            useRightHJCnodeLabel = model.m_properties["CalibrationParameters"]["RHJC_node"]
            useRightKJCnodeLabel = model.m_properties["CalibrationParameters"]["RKJC_node"]
            useRightAJCnodeLabel = model.m_properties["CalibrationParameters"]["RAJC_node"]
            useRotateRightThighFlag = True
            useRotateLeftThighFlag = False


        # ----  Calibration

        logging.debug("Calibration2Dof --%s --- final Calibration"%(side) )
        logging.debug(" node (LHJC) => %s" %(useLeftHJCnodeLabel))
        logging.debug(" node (LKJC) => %s" %(useLeftKJCnodeLabel))
        logging.debug(" node (LAJC) => %s" %(useLeftAJCnodeLabel))
        logging.debug(" rotated Left Thigh => %s" %(str(useRotateLeftThighFlag)))
        logging.debug("-opposite side-" )
        logging.debug(" node (RHJC) => %s" %(useRightHJCnodeLabel))
        logging.debug(" node (RKJC) => %s" %(useRightKJCnodeLabel))
        logging.debug(" node (RAJC) => %s" %(useRightAJCnodeLabel))
        logging.debug(" rotated Right Thigh => %s" %(str(useRotateRightThighFlag)))


        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           useLeftHJCnode=useLeftHJCnodeLabel, useRightHJCnode=useRightHJCnodeLabel,
                           useLeftKJCnode=useLeftKJCnodeLabel, useLeftAJCnode=useLeftAJCnodeLabel,
                           useRightKJCnode=useRightKJCnodeLabel, useRightAJCnode=useRightAJCnodeLabel,
                           leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                           markerDiameter=markerDiameter,
                           RotateLeftThighFlag = useRotateLeftThighFlag,
                           RotateRightThighFlag = useRotateRightThighFlag).compute()


        logging.warning("model updated with a  %s knee calibrated with 2Dof method" %(side))
        # ----------------------EXPORT-------------------------------------------
        # overwrite func file
        btkTools.smartWriter(acqFunc, str(DATA_PATH+trial[:-4]+"-modelled.c3d"))


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
