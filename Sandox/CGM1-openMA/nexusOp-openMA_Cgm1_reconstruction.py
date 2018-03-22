# -*- coding: utf-8 -*-
"""
 
"""

import sys
import pdb
import logging
import argparse


# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)


# vicon
pyCGM2.CONFIG.addNexusPythonSdk()
import ViconNexus

# openMA
pyCGM2.CONFIG.addOpenma()
import ma.io
import ma.body
   
# pyCGM2 libraries   
from  pyCGM2.Tools  import trialTools   
from pyCGM2 import  smartFunctions 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CGM')
    parser.add_argument('--Calibration', action='store_true', help='calibration' )
    parser.add_argument('--staticFile', default="", type=str)
    parser.add_argument('-l','--leftFlatFoot', action='store_true', help='left flat foot flag' )
    parser.add_argument('-r','--rightFlatFoot', action='store_true', help='right flat foot flag' )
    parser.add_argument('-p','--processing', action='store_true', help='enable plot' )
    parser.add_argument('--markerDiameter', default=14, type=float)
    parser.add_argument('--pointSuffix', default="", type=str)
    parser.add_argument('--author', default="Schwartz2008", type=str)
    parser.add_argument('--modality', default="Free", type=str)    
    
    args = parser.parse_args()
    
    logging.info(args)
    
    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected() 

        
    if NEXUS_PYTHON_CONNECTED: # run operation
        
        #---- INPUTS------
        if args.Calibration:
            calibrateFilenameLabelledNoExt = None  
        else:
            calibrateFilenameLabelledNoExt = args.staticFile

        flag_leftFlatFoot =  args.leftFlatFoot 
        flag_rightFlatFoot = args.rightFlatFoot 
        markerDiameter =  args.markerDiameter 
        if  args.pointSuffix == "":
            pointSuffix = ""
        else:
            pointSuffix = args.pointSuffix  

        enableProcessing = args.processing
        normativeDataInput = str(args.author+"_"+ args.modality) #"Schwartz2008_VeryFast"
       
               
        #---- DATA ----        
        DATA_PATH, reconstructFilenameLabelledNoExt = pyNEXUS.GetTrialName()
        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        if calibrateFilenameLabelledNoExt is None:
            logging.warning("Static Processing")
            staticProcessing = True
            calibrateFilenameLabelled = reconstructFilenameLabelled
        else:
            staticProcessing = False
            calibrateFilenameLabelled = calibrateFilenameLabelledNoExt + ".c3d"

        logging.info( "data Path: "+ DATA_PATH )   
        logging.info( "calibration file: "+ calibrateFilenameLabelled)
        logging.info( "reconstruction file: "+ reconstructFilenameLabelled )     

        
        # ----- SUBJECT --------
        subjects = pyNEXUS.GetSubjectNames()
        subject =   subjects[0]  
        logging.info( "Subject name : " + subject)  

        Parameters = pyNEXUS.GetSubjectParamNames(subject)

        sub = ma.Subject(str(subject)) 
        sub.setProperty("markerDiameter",ma.Any(markerDiameter))
        sub.setProperty("leftLegLength",ma.Any(pyNEXUS.GetSubjectParamDetails( subject, "LeftLegLength")[0]))
        sub.setProperty("leftKneeWidth",ma.Any(pyNEXUS.GetSubjectParamDetails( subject, "LeftKneeWidth")[0]))
        sub.setProperty("leftAnkleWidth",ma.Any(pyNEXUS.GetSubjectParamDetails( subject, "LeftAnkleWidth")[0]))
        sub.setProperty("rightLegLength",ma.Any(pyNEXUS.GetSubjectParamDetails( subject, "RightLegLength")[0]))
        sub.setProperty("rightKneeWidth",ma.Any(pyNEXUS.GetSubjectParamDetails( subject, "RightKneeWidth")[0]))
        sub.setProperty("rightAnkleWidth",ma.Any(pyNEXUS.GetSubjectParamDetails( subject, "RightAnkleWidth")[0]))
        sub.setProperty("mass",ma.Any(pyNEXUS.GetSubjectParamDetails( subject, "Bodymass")[0]))
        sub.setProperty("height",ma.Any(pyNEXUS.GetSubjectParamDetails( subject, "Height")[0]))
        
        # ----- STATIC CALIBRATION --------
        # reader
        staticNode = ma.io.read(str(DATA_PATH+calibrateFilenameLabelled))
    
        # calibration
        cgm1_static = ma.body.PluginGait(ma.body.Region_Lower, ma.body.Side_Both,ma.body.PluginGait.Variant_Basic)
        cgm1_static.setProperty("gravity",ma.Any([0.0,0.0,-9.81]))
        cgm1_static.setProperty("leftFootFlatEnabled",ma.Any(flag_leftFlatFoot))
        cgm1_static.setProperty("rightFootFlatEnabled",ma.Any(flag_rightFlatFoot))
        
        cgm1_static.calibrate(staticNode,sub)
        
        # ----- GAIT RECONSTRUCTION --------
        # reader    
        dynamicNode = ma.io.read(str(DATA_PATH + reconstructFilenameLabelled))
        dynamicTrial = dynamicNode.findChild(ma.T_Trial) # get the first one !
    
        # kinematics and Kinetics Calculation
        cgm1_gait = ma.body.reconstruct(cgm1_static,dynamicNode)
     
        #---- EXTRACT KINEMATICS AND KINETICS -----
        kinematics  = ma.body.extract_joint_kinematics(cgm1_gait)
        trialTools.renameOpenMAtoVicon(kinematics)
        # append new parameters to the gait trial    
        trialTools.addTimeSequencesToTrial(dynamicTrial,kinematics)

        if not staticProcessing:
            kinetics = ma.body.extract_joint_kinetics(cgm1_gait)
            trialTools.renameOpenMAtoVicon(kinetics)

            trialTools.addTimeSequencesToTrial(dynamicTrial,kinetics)
    
        # add property
        dynamicTrial.setProperty('MODEL:NAME',"CGM1")
        dynamicTrial.setProperty('MODEL:PROCESSOR',"openMA")
        
        # ----- WRITER --------
        if ma.io.write(dynamicNode,str(DATA_PATH + reconstructFilenameLabelled[:-4] + "_openmaCGM1.c3d")):
            logging.info( "file ( %s) reconstructed ( suffix : _openmaCGM1.c3d) " % (reconstructFilenameLabelled) )                 

        # -----------CGM PROCESSING--------------------
        if enableProcessing:

            # infos        
            model= None 
            subject=None       
            experimental=None
    
            if staticProcessing:
                
                # temporal static angle and static angle profile
                smartFunctions.staticProcessing_cgm1(str(reconstructFilenameLabelled[:-4] + "_openmaCGM1.c3d"), DATA_PATH,
                                                     model,  subject, experimental,
                                                     pointLabelSuffix = pointSuffix)          
            else:
     
    
                normativeData = { "Author": normativeDataInput[:normativeDataInput.find("_")],"Modality": normativeDataInput[normativeDataInput.find("_")+1:]} 
                             
                # ----PROCESSING-----
                smartFunctions.gaitProcessing_cgm1 (str(reconstructFilenameLabelled[:-4] + "_openmaCGM1.c3d"), DATA_PATH,
                                       model,  subject, experimental,
                                       pointLabelSuffix = pointSuffix,
                                       plotFlag= True, 
                                       exportBasicSpreadSheetFlag = False,
                                       exportAdvancedSpreadSheetFlag = False,
                                       exportAnalysisC3dFlag = False,
                                       consistencyOnly = True,
                                       normativeDataDict = normativeData)
              
    else: 
        logging.error(" Nexus Not Connected")        
