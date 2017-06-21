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

    parser = argparse.ArgumentParser(description='SARA Knee Functional Calibration')
    parser.add_argument('--version', type=str, help='version of cgm2.i',default="2.3")
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------PATH + FILE ------------------------------------

        if DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.3\\Knee Calibration\\" 
            reconstructedFilenameLabelledNoExt = "Left Knee"
            NEXUS.OpenTrial( str(DATA_PATH+reconstructedFilenameLabelledNoExt), 30 )
            args.version=="2.3"

        
        else:
            DATA_PATH, reconstructedFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ reconstructFilenameLabelled)

        # --------------------CHOICE -----------------------------
        if args.version=="2.3":
            modelName = "CGM2_3-pyCGM2.model"
            globalSettingName = "CGM2_3-pyCGM2.settings"
        elif args.version=="2.3e":        
            modelName = "CGM2_3-Expert-pyCGM2.model"
            globalSettingName = "CGM2_3-Expert-pyCGM2.settings"
        elif args.version=="2.4":        
            modelName = "CGM2_4-pyCGM2.model"
            globalSettingName = "CGM2_4-pyCGM2.settings"
        elif args.version=="2.4e":        
            modelName = "CGM2_4-Expert-pyCGM2.model"
            globalSettingName = "CGM2_4-Expert-pyCGM2.settings"
        else:
            raise Exception ("[pyCGM2] version of cgm2.3+ dont recognize ( recognized versions are 2.3 or 2.3e 2.4 and 2.4e)")


        # --------------------GLOBAL SETTINGS ------------------------------
        # global setting ( in user/AppData)
        inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+globalSettingName)).read(),object_pairs_hook=OrderedDict)
       

        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.ckeckActivatedSubject(NEXUS,subjects,"LASI")
        logging.info(  "Subject name : " + subject  )
            
        # --------------------pyCGM2 MODEL ------------------------------
        if not os.path.isfile(DATA_PATH + subject + "-"+modelName):
            raise Exception ("%s-%s file doesn't exist. Run Calibration operation"%(subject,modelName))
        else:
            f = open(DATA_PATH + subject + "-"+ modelName, 'r')
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
        if args.version=="2.3" or args.version=="2.3e": 
            if os.path.isfile( DATA_PATH + "CGM2-3.translators"):
               logging.warning("local translator found")
               sessionTranslators = json.loads(open(DATA_PATH + "CGM2-3.translators").read(),object_pairs_hook=OrderedDict)
               translators = sessionTranslators["Translators"]
            else:
               translators = inputs["Translators"]            
        elif args.version=="2.4" or args.version=="2.4e":
            if os.path.isfile( DATA_PATH + "CGM2-4.translators"):
               logging.warning("local translator found")
               sessionTranslators = json.loads(open(DATA_PATH + "CGM2-4.translators").read(),object_pairs_hook=OrderedDict)
               translators = sessionTranslators["Translators"]
            else:
               translators = inputs["Translators"]     
        else:
            raise Exception ("[pyCGM2] version of cgm dont recognize ( recognized versions are 2.3 or 2.3e 2.4 and 2.4e)")
            
        # --------------------------ACQ WITH TRANSLATORS --------------------------------------

        # --- btk acquisition ----
        acqFunc = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

        btkTools.checkMultipleSubject(acqFunc)
        acqFunc =  btkTools.applyTranslators(acqFunc,translators)
         

        # motion side of the lower limb 
        side = detectSide(acqFunc,"LANK","RANK")
        logging.info("Detected motion side : %s" %(side) )

        if side == "Left":
            thigh_markers = model.getSegment("Left Thigh").m_tracking_markers
            shank_markers = model.getSegment("Left Shank").m_tracking_markers
 
        elif side == "Right":
            thigh_markers = model.getSegment("Right Thigh").m_tracking_markers
            shank_markers = model.getSegment("Right Shank").m_tracking_markers

        validFrames,vff,vlf = btkTools.findValidFrames(acqFunc,thigh_markers+shank_markers) 



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

        if os.path.isfile(str(DATA_PATH + subject + "-"+modelName)):
            logging.warning("previous model removed")
            os.remove(str(DATA_PATH + subject + "-"+modelName))

        modelFile = open(str(DATA_PATH + subject + "-"+modelName), "w")
        cPickle.dump(model, modelFile)
        modelFile.close()
        logging.warning("model updated with SARA knee Joint centre")
            
            

        # ----------------------VICON INTERFACE-------------------------------------------

        #--- update mp
        viconInterface.updateNexusSubjectMp(NEXUS,model,subject)

        nexusTools.appendBones(NEXUS,subject,acqFunc,"LFE", model.getSegment("Left Thigh"),OriginValues = acqFunc.GetPoint("LKJC").GetValues() )

        #--- Add modelled markers
#        Or_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KneeFlexionOri")
#        axis_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KneeFlexionAxis")
#        Or_inShank = model.getSegment(distalSegmentLabel).getReferential("TF").getNodeTrajectory("KneeFlexionOri")
#        axis_inShank = model.getSegment(distalSegmentLabel).getReferential("TF").getNodeTrajectory("KneeFlexionAxis")
#               
#        
#        
#        btkTools.smartAppendPoint(acqFunc,side+"_KneeFlexA_inThigh",Or_inThigh)
#        btkTools.smartAppendPoint(acqFunc,side+"_KneeFlexB_inThigh",axis_inThigh) 
#       
#        btkTools.smartAppendPoint(acqFunc,side+"_KneeFlexA_inShank",Or_inShank)
#        btkTools.smartAppendPoint(acqFunc,side+"_KneeFlexB_inShank",axis_inShank) 
#
#        
#        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KneeFlexA_inThigh", acqFunc)
#        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KneeFlexB_inThigh", acqFunc)     
#        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KneeFlexA_inShank", acqFunc)
#        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KneeFlexB_inShank", acqFunc)     


        meanOr_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KJC_Sara")
        meanAxis_inThigh = model.getSegment(proximalSegmentLabel).getReferential("TF").getNodeTrajectory("KJC_SaraAxis")

        btkTools.smartAppendPoint(acqFunc,side+"_KJC_Sara",meanOr_inThigh)
        btkTools.smartAppendPoint(acqFunc,side+"_KJC_SaraAxis",meanAxis_inThigh) 

        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KJC_Sara", acqFunc)
        nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,side+"_KJC_SaraAxis", acqFunc) 

        #btkTools.smartWriter(acqFunc, "acqFunc-test.c3d")
        

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")