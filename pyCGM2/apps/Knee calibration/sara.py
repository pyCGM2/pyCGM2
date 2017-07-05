# -*- coding: utf-8 -*-
import sys
import pdb
import logging
import matplotlib.pyplot as plt
import argparse
import json
import os
from collections import OrderedDict
from shutil import copyfile
import cPickle
import numpy as np

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

    
# pyCGM2 libraries    
from pyCGM2.Tools import btkTools,nexusTools
from pyCGM2.Model.CGM2 import cgm2, modelFilters, modelDecorator
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

    parser = argparse.ArgumentParser(description='SARA Functional Knee Calibration')
    args = parser.parse_args()

    
    # --------------------SESSION SETTINGS ------------------------------
    DATA_PATH =os.getcwd()+"\\"
    infoSettings = json.loads(open('pyCGM2.info').read(),object_pairs_hook=OrderedDict)

    # --------------------CONFIGURATION ------------------------------
    # NA

    # --------------------pyCGM2 MODEL - INIT ------------------------------
    
    if not os.path.isfile(DATA_PATH +  "pyCGM2-INIT.model"):
        raise Exception ("pyCGM2-INIT.model file doesn't exist. Run Calibration operation")
    else:
        f = open(DATA_PATH  + 'pyCGM2-INIT.model', 'r')
        model = cPickle.load(f)
        f.close()

    logging.info("loaded model : %s" %(model.version ))

    # --------------------CHECKING ------------------------------       
    if model.version in ["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e"] :
        raise Exception ("Can t use SARA method with your model %s"%(model.version))
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
    
    # --------------------------SESSION INFOS ------------------------------------
    
    #  translators management
    if model.version in  ["CGM2.3","CGM2.3e"]:
        translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM2-3.translators")
    elif model.version in  ["CGM2.4","CGM2.4e"]:
        translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM2-4.translators")
    if not translators:
       translators = inputs["Translators"]   
            
    # --------------------------ACQ WITH TRANSLATORS --------------------------------------
    funcTrials = infoSettings["Modelling"]["KneeCalibrationTrials"]["SaraTrials"]

    for trial in funcTrials:

        # --- btk acquisition ----
        acqFunc = btkTools.smartReader(str(DATA_PATH + trial))
        acqFunc =  btkTools.applyTranslators(acqFunc,translators)
         
        # motion side of the lower limb 
        side = detectSide(acqFunc,"LANK","RANK")
        logging.info("Detected motion side : %s" %(side) )

        scp=modelFilters.StaticCalibrationProcedure(model)
        
        if model.version in  ["CGM2.3","CGM2.3e","CGM2.3","CGM2.4e"]:
            if side == "Left":
                thigh_markers = model.getSegment("Left Thigh").m_tracking_markers
                shank_markers = model.getSegment("Left Shank").m_tracking_markers
 
            elif side == "Right":
                thigh_markers = model.getSegment("Right Thigh").m_tracking_markers
                shank_markers = model.getSegment("Right Shank").m_tracking_markers

            validFrames,vff,vlf = btkTools.findValidFrames(acqFunc,thigh_markers+shank_markers) 

            proximalSegmentLabel=str(side+" Thigh")
            distalSegmentLabel=str(side+" Shank")        
              
            # Motion of only left 
            modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,pyCGM2Enums.motionMethod.Sodervisk)
            modMotion.segmentalCompute([proximalSegmentLabel,distalSegmentLabel])
        
            # decorator
            modelDecorator.KneeCalibrationDecorator(model).sara(side)

            # --------------------------NEW CALIBRATION OF THE STATIC File---------
            # WRNING : STATIC FILE From member of model !!! ()
    
            staticFilename = model.m_staticFilename
            acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilename))
            btkTools.checkMultipleSubject(acqStatic)
            acqStatic =  btkTools.applyTranslators(acqStatic,translators)
               
            if side == "Left":        
                modelFilters.ModelCalibrationFilter(scp,acqStatic,model, useLeftKJCnode="KJC_Sara").compute()
            elif side == "Right":        
                modelFilters.ModelCalibrationFilter(scp,acqStatic,model, useRightKJCnode="KJC_Sara").compute()
    
            # ----------------------SAVE-------------------------------------------
            if os.path.isfile(DATA_PATH + "pyCGM2.model"):
                logging.warning("previous model removed")
                os.remove(DATA_PATH +  "-pyCGM2.model")
    
            modelFile = open(DATA_PATH + "pyCGM2.model", "w")
            cPickle.dump(model, modelFile)
            modelFile.close()
            logging.warning("model updated with a  %s knee calibrated with SARA method" %(side))
            
            
            # ----------------------EXPORT-------------------------------------------
            # add modelled markers
            meanOr_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KJC_Sara")
            meanAxis_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KJC_SaraAxis")

            btkTools.smartAppendPoint(acqFunc,side+"_KJC_Sara",meanOr_inThigh)
            btkTools.smartAppendPoint(acqFunc,side+"_KJC_SaraAxis",meanAxis_inThigh) 

            btkTools.smartWriter(acqFunc, str(DATA_PATH+trial))
