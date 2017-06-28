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
from pyCGM2.Model.CGM2 import cgm, modelFilters, modelDecorator
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

    parser = argparse.ArgumentParser(description='dynaKAD Knee Calibration for cgm1 markerset')
    parser.add_argument('--version', type=str, help='version of cgm',default="1")
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------PATH + FILE ------------------------------------

        if DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\dynaKAD\\" 
            reconstructedFilenameLabelledNoExt = "MRI-US-01, 2008-08-08, 3DGA 14"
            NEXUS.OpenTrial( str(DATA_PATH+reconstructedFilenameLabelledNoExt), 30 )

        
        else:
            DATA_PATH, reconstructedFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ reconstructFilenameLabelled)

        # --------------------CHOICE -----------------------------
        if args.version=="1":
            modelName = "CGM1-pyCGM2.model"
            globalSettingName = "CGM1-pyCGM2.settings"
        elif args.version=="1.1":        
            modelName = "CGM1_1-pyCGM2.model"
            globalSettingName = "CGM1_1-pyCGM2.settings"
        elif args.version=="2.1":        
            modelName = "CGM2_1-pyCGM2.model"
            globalSettingName = "CGM2_1-pyCGM2.settings"
        elif args.version=="2.2":        
            modelName = "CGM2_2-pyCGM2.model"
            globalSettingName = "CGM2_2-pyCGM2.settings"
        elif args.version=="2.2e":        
            modelName = "CGM2_2-Expert-pyCGM2.model"
            globalSettingName = "CGM2_2-Expert-pyCGM2.settings"
        else:
            raise Exception ("[pyCGM2] version of cgm not recognized ( available choices are : 1 or 1.1 or 2.1 or 2.2 or 2.2e)")

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
        if os.path.isfile( DATA_PATH + "CGM1.translators"):
           logging.warning("local translator found")
           sessionTranslators = json.loads(open(DATA_PATH + "CGM1.translators").read(),object_pairs_hook=OrderedDict)
           translators = sessionTranslators["Translators"]
        else:
           translators = inputs["Translators"]            
 
            
        # --------------------------ACQ WITH TRANSLATORS --------------------------------------

        # --- btk acquisition ----
        acqFunc = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

        btkTools.checkMultipleSubject(acqFunc)
        acqFunc =  btkTools.applyTranslators(acqFunc,translators)
         
        # static calibration procedure 
        scp=modelFilters.StaticCalibrationProcedure(model)

        # motion  
        side = detectSide(acqFunc,"LANK","RANK")
        logging.info("Detected motion side : %s" %(side) )

        validFrames,vff,vlf = btkTools.findValidFrames(acqFunc,cgm.CGM1LowerLimbs.MARKERS) 


        modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()

        # calibration decorators
        if side == "Left":
            modelDecorator.KneeCalibrationDecorator(model).calibrate2dof("Left")
        elif side == "Right":
            modelDecorator.KneeCalibrationDecorator(model).calibrate2dof("Right")


        # --------------------------NEW CALIBRATION OF THE STATIC File---------
        # WRNING : STATIC FILE From member of model !!! ()

        staticFilename = model.m_staticFilename
        acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilename))
        btkTools.checkMultipleSubject(acqStatic)
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)
        
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 



        # ----------------------SAVE-------------------------------------------
                   
        if os.path.isfile(str(DATA_PATH + subject + "-"+modelName)):
            logging.warning("previous model removed")
            os.remove(str(DATA_PATH + subject + "-"+modelName))

        modelFile = open(str(DATA_PATH + subject + "-"+modelName), "w")
        cPickle.dump(model, modelFile)
        modelFile.close()
        logging.warning("model updated with Knee dynaKad offset")

           

        # ----------------------VICON INTERFACE-------------------------------------------

        #--- update mp
        viconInterface.updateNexusSubjectMp(NEXUS,model,subject)


        #btkTools.smartWriter(acqFunc, "acqFunc-test.c3d")
        

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")