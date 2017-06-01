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

# vicon nexus
import ViconNexus
    
# pyCGM2 libraries    
from pyCGM2.Tools import btkTools,nexusTools
from pyCGM2.Model.CGM2 import cgm2, modelFilters, modelDecorator
import pyCGM2.enums as pyCGM2Enums


def detectSide(acq,left_markerLabel,right_markerLabel):

    flag,vff,vlf = btkTools.findValidFrames(acq,[left_markerLabel,right_markerLabel])
    
    left = acq.GetPoint(left_markerLabel).GetValues()[vff:vlf,2]
    right = acq.GetPoint(right_markerLabel).GetValues()[vff:vlf,2]

    side = "Left" if np.max(left)>np.max(right) else "Right"

    return side

    
if __name__ == "__main__":
   
    plt.close("all")
    DEBUG = False

#    parser = argparse.ArgumentParser(description='Gait Processing')
#    parser.add_argument('--pointSuffix', type=str, help='force suffix')
#    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------PATH + FILE ------------------------------------

        if DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.3\\Knee Calibration\\" 
            reconstructedFilenameLabelledNoExt = "Left Knee"
            NEXUS.OpenTrial( str(DATA_PATH+reconstructedFilenameLabelledNoExt), 30 )

        else:
            DATA_PATH, reconstructedFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ reconstructFilenameLabelled)

        # --------------------GLOBAL SETTINGS ------------------------------
        # global setting ( in user/AppData)
        inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_3-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)


        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.ckeckActivatedSubject(NEXUS,subjects,"LASI")
        logging.info(  "Subject name : " + subject  )
            
        # --------------------pyCGM2 MODEL ------------------------------
        if not os.path.isfile(DATA_PATH + subject + "-CGM2_3-pyCGM2.model"):
            raise Exception ("%s-CGM2_3-pyCGM2.model file doesn't exist. Run Calibration operation"%subject)
        else:
            f = open(DATA_PATH + subject + '-CGM2_3-pyCGM2.model', 'r')
            model = cPickle.load(f)
            f.close()
        
        # --------------------------SESSION INFOS -----------------------------
        # info file
        if not os.path.isfile( DATA_PATH + subject+"-pyCGM2.info"):
            copyfile(str(pyCGM2.CONFIG.PYCGM2_SESSION_SETTINGS_FOLDER+"pyCGM2.info"), str(DATA_PATH + subject+"-pyCGM2.info"))
            logging.warning("Copy of pyCGM2.info from pyCGM2 Settings folder")
            infoSettings = json.loads(open(DATA_PATH +subject+'-pyCGM2.info').read(),object_pairs_hook=OrderedDict)
        else:
            infoSettings = json.loads(open(DATA_PATH +subject+'-pyCGM2.info').read(),object_pairs_hook=OrderedDict)


        #  translators management 
        if os.path.isfile( DATA_PATH + "CGM2-3.translators"):
           logging.warning("local translator found")
           sessionTranslators = json.loads(open(DATA_PATH + "CGM2-3.translators").read(),object_pairs_hook=OrderedDict)
           translators = sessionTranslators["Translators"]
        else:
           translators = inputs["Translators"]            
            
            
        # --------------------------ACQ WITH TRANSLATORS --------------------------------------

        # --- btk acquisition ----
        acqFunc = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

        btkTools.checkMultipleSubject(acqFunc)
        acqFunc =  btkTools.applyTranslators(acqFunc,translators)
        validFrames,vff,vlf = btkTools.findValidFrames(acqFunc,cgm2.CGM2_3LowerLimbs.MARKERS)

        # motion side of the lower limb 
        side = detectSide(acqFunc,"LANK","RANK")
        logging.info("Detected motion side : %s" %(side) )

        proximalSegmentLabel=str(side+" Thigh")
        distalSegmentLabel=str(side+" Shank")        

        scp=modelFilters.StaticCalibrationProcedure(model)
               
        # Motion of only left 
        modMotionLeftKnee=modelFilters.ModelMotionFilter(scp,acqFunc,model,pyCGM2Enums.motionMethod.Sodervisk)
        modMotionLeftKnee.segmentalCompute([proximalSegmentLabel,distalSegmentLabel])
        
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
        if os.path.isfile(DATA_PATH + subject + "-CGM2_3-pyCGM2.model"):
            logging.warning("previous model removed")
            os.remove(DATA_PATH + subject + "-CGM2_3-pyCGM2.model")

        modelFile = open(DATA_PATH + subject+"-CGM2_3-pyCGM2.model", "w")
        cPickle.dump(model, modelFile)
        modelFile.close()
        logging.warning("model updated with SARA knee Joint centre")



        # ----------------------VICON INTERFACE-------------------------------------------

        Or_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KneeFlexionOri")
        axis_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KneeFlexionAxis")
        Or_inShank = model.getSegment(distalSegmentLabel).getReferential("TF").getNodeTrajectory("KneeFlexionOri")
        axis_inShank = model.getSegment(distalSegmentLabel).getReferential("TF").getNodeTrajectory("KneeFlexionAxis")
        
        btkTools.smartAppendPoint(acqFunc,side+"_KneeFlexA_inThigh",Or_inThigh)
        btkTools.smartAppendPoint(acqFunc,side+"_KneeFlexB_inThigh",axis_inThigh) 
       
        btkTools.smartAppendPoint(acqFunc,side+"_KneeFlexA_inShank",Or_inShank)
        btkTools.smartAppendPoint(acqFunc,side+"_KneeFlexB_inShank",axis_inShank) 
        #
        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KneeFlexA_inThigh", acqFunc)
        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KneeFlexB_inThigh", acqFunc)     
        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KneeFlexA_inShank", acqFunc)
        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KneeFlexB_inShank", acqFunc)     

        #btkTools.smartWriter(acqFunc, "acqFunc-test.c3d")
        

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")