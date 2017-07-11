# -*- coding: utf-8 -*-
import sys
import pdb
import logging
import matplotlib.pyplot as plt
import argparse
import json
import os
from collections import OrderedDict
import cPickle
import numpy as np

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

    
# pyCGM2 libraries    
from pyCGM2.Tools import btkTools,nexusTools
from pyCGM2.Model.CGM2 import cgm,cgm2, modelFilters, modelDecorator
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Utils import fileManagement

from pyCGM2 import viconInterface

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
    args = parser.parse_args()

    # --------------------SESSION SETTINGS ------------------------------
    DATA_PATH =os.getcwd()+"\\"
    infoSettings = json.loads(open('pyCGM2.info').read(),object_pairs_hook=OrderedDict)

    # --------------------CONFIGURATION ------------------------------
    # NA

    # --------------------pyCGM2 MODEL------------------------------
    
    if not os.path.isfile(DATA_PATH +  "pyCGM2.model"):
        raise Exception ("pyCGM2.model file doesn't exist. Run Calibration operation")
    else:
        f = open(DATA_PATH  + 'pyCGM2.model', 'r')
        model = cPickle.load(f)
        f.close()

    logging.info("loaded model : %s" %(model.version ))

    # --------------------GLOBAL SETTNGS ------------------------------
       
    if model.version == "CGM1.0":
           inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM1-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
    elif model.version == "CGM1.1":
           inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM1_1-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
    elif model.version == "CGM2.1":
           inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_1-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
    elif model.version == "CGM2.2":
           inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_2-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
    elif model.version == "CGM2.2e":
           inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_2-Expert-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
    elif model.version == "CGM2.3":
           inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_3-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
    elif model.version == "CGM2.3e":
           inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_3-Expert-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
    elif model.version == "CGM2.4":
           inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_4-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
    elif model.version == "CGM2.4e":
           inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_4-Expert-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
    else:
        raise Exception ("model version not found [contact admin]")
        
    # --------------------------TRANSLATORS ------------------------------------

    #  translators management
    if model.version in  ["CGM1.0"]:
        translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM1.translators")
    elif model.version in  ["CGM1.1","CGM2.1","CGM2.2","CGM2.2e"]:
        translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM1-1.translators")
    elif model.version in  ["CGM2.3","CGM2.3e"]:
        translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM2-3.translators")
    elif model.version in  ["CGM2.4","CGM2.4e"]:
        translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM2-4.translators")

    if not translators:
       translators = inputs["Translators"]        
            
    # --------------------------ACQ WITH TRANSLATORS --------------------------------------
    funcTrials = infoSettings["Modelling"]["KneeCalibrationTrials"]["Calibration2Dof"]

    for it in funcTrials:
        
        trial = it["Trial"]

        # --- btk acquisition ----
        acqFunc = btkTools.smartReader(str(DATA_PATH + trial))
        acqFunc =  btkTools.applyTranslators(acqFunc,translators)
        
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



        btkTools.smartWriter(acqStatic, "acqStatic0-test.c3d")
        

        if model.version in  ["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e"]:
            
            modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,pyCGM2Enums.motionMethod.Determinist)
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
        modelDecorator.KneeCalibrationDecorator(model).calibrate2dof(side)


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

    if os.path.isfile(DATA_PATH +  "pyCGM2.model"):
        logging.warning("previous model removed")
        os.remove(DATA_PATH +  "pyCGM2.model")

    modelFile = open(DATA_PATH + "pyCGM2.model", "w")
    cPickle.dump(model, modelFile)
    modelFile.close()
    
    
        
        

        

 
