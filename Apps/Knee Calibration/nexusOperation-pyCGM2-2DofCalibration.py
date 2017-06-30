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

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------PATH + FILE ------------------------------------

        if DEBUG:
            # for CGM1 to CGM2.2
#            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\knee calibration\\CGM1-calibration2Dof\\" 
#            reconstructedFilenameLabelledNoExt = "Left Knee"

            # for CGM2.3 to ...
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\knee calibration\\CGM2.3-calibration2Dof\\" 
            reconstructedFilenameLabelledNoExt = "Left Knee"

 
            NEXUS.OpenTrial( str(DATA_PATH+reconstructedFilenameLabelledNoExt), 30 )
        
        else:
            DATA_PATH, reconstructedFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "reconstructed file: "+ reconstructFilenameLabelled)

        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.ckeckActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        if not os.path.isfile(DATA_PATH + subject + "-pyCGM2.model"):
            raise Exception ("%s-pyCGM2.model file doesn't exist. Run Calibration operation"%subject)
        else:
            f = open(DATA_PATH + subject + '-pyCGM2.model', 'r')
            model = cPickle.load(f)
            f.close()

        logging.info("loaded model : %s" %(model.version ))
       
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
               inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_3-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
        elif model.version == "CGM2.4":
               inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_3-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)
        elif model.version == "CGM2.4e":
               inputs = json.loads(open(str(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH+"CGM2_3-pyCGM2.settings")).read(),object_pairs_hook=OrderedDict)

        else:
            raise Exception ("model version not found [contact admin]")
        
        # --------------------------SESSION INFOS ------------------------------------
        # info file
        infoSettings = fileManagement.manage_pycgm2SessionInfos(DATA_PATH,subject)
        
        #  translators management
        if model.version in  ["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e"]:
            translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM1.translators")
        elif model.version in  ["CGM2.3","CGM2.3e"]:
            translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM2-3.translators")
        elif model.version in  ["CGM2.4","CGM2.4e"]:
            translators = fileManagement.manage_pycgm2Translators(DATA_PATH,"CGM2-4.translators")

        if not translators:
           translators = inputs["Translators"]        
 
            
        # --------------------------ACQ WITH TRANSLATORS --------------------------------------

        # --- btk acquisition ----
        acqFunc = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

        btkTools.checkMultipleSubject(acqFunc)
        acqFunc =  btkTools.applyTranslators(acqFunc,translators)
        
        # motion  
        side = detectSide(acqFunc,"LANK","RANK")
        logging.info("Detected motion side : %s" %(side) )
        
        if model.version in  ["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.2e"]:
            validFrames,vff,vlf = btkTools.findValidFrames(acqFunc,cgm.CGM1LowerLimbs.MARKERS)
         
        # static calibration procedure
        scp=modelFilters.StaticCalibrationProcedure(model)
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
              
            # Motion of only left 
            modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,pyCGM2Enums.motionMethod.Sodervisk)
            modMotion.segmentalCompute([proximalSegmentLabel,distalSegmentLabel])
        

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
        if os.path.isfile(DATA_PATH + subject + "-pyCGM2.model"):
            logging.warning("previous model removed")
            os.remove(DATA_PATH + subject + "-pyCGM2.model")

        modelFile = open(DATA_PATH + subject+"-pyCGM2.model", "w")
        cPickle.dump(model, modelFile)
        modelFile.close()
        logging.warning("model updated with a  %s knee calibrated with 2Dof method" %(side))
       
       # ----------------------VICON INTERFACE-------------------------------------------

        #--- update mp
        viconInterface.updateNexusSubjectMp(NEXUS,model,subject)


        #btkTools.smartWriter(acqFunc, "acqFunc-test.c3d")
        

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")