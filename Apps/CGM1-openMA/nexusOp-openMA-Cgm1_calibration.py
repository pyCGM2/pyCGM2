# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:05:09 2016

@author: aaa34169
"""

import sys
import pdb
import logging


# pyCGM2 settings
import pyCGM2
pyCGM2.pyCGM2_CONFIG.setLoggingLevel(logging.INFO)

# vicon
pyCGM2.pyCGM2_CONFIG.addNexusPythonSdk()
import ViconNexus

# openMA
pyCGM2.pyCGM2_CONFIG.addOpenma()
import ma.io
import ma.body

#pyCGM2 libraries
import pyCGM2.Core.Model.openmaLib as openmaLib
from  pyCGM2.Core.Tools  import trialTools



if __name__ == "__main__":
     
    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected() 

        
    if NEXUS_PYTHON_CONNECTED: # run Operation
        
        #---- INPUTS------   
        flag_leftFlatFoot =  bool(int(sys.argv[1]))
        flag_rightFlatFoot =  bool(int(sys.argv[2]))
        markerDiameter = float(sys.argv[3])


        #---- DATA ------           
        DATA_PATH, calibrateFilenameLabelledNoExt = pyNEXUS.GetTrialName()
        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"
        reconstructFilenameLabelled = calibrateFilenameLabelled 
       
        logging.info( "data Path: "+ DATA_PATH )   
        logging.info( "calibration file: "+ calibrateFilenameLabelled)
        logging.info( "reconstruction file: "+ reconstructFilenameLabelled )      
        

        # ----- SUBJECT --------
        subjects = pyNEXUS.GetSubjectNames()
        subject =   subjects[0]   
        logging.info(  "Subject name : " + subject  )

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
        statictrial = ma.io.read(str(DATA_PATH+calibrateFilenameLabelled))
    
        # calibration
        cgm1_static = ma.body.PluginGait(ma.body.Region_Lower, ma.body.Side_Both,ma.body.PluginGait.Variant_Basic)
        cgm1_static.setProperty("gravity",ma.Any([0.0,0.0,-9.81]))
        cgm1_static.setProperty("leftFootFlatEnabled",ma.Any(flag_leftFlatFoot))
        cgm1_static.setProperty("rightFootFlatEnabled",ma.Any(flag_rightFlatFoot))
        
        cgm1_static.calibrate(statictrial,sub)
        
        # ----- GAIT RECONSTRUCTION --------
        # reader    
        dynamicNode = ma.io.read(str(DATA_PATH + reconstructFilenameLabelled))
        dynamicTrial = dynamicNode.findChild(ma.T_Trial) # get the first one !   

        # kinematics and Kinetics Calculation
        cgm1_gait = ma.body.reconstruct(cgm1_static,dynamicTrial)
     
        #---- EXTRACT KINEMATICS AND KINETICS -----
        kinematics  = ma.body.extract_joint_kinematics(cgm1_gait)
        openmaLib.renameOpenMAtoVicon(kinematics)

        kinetics = ma.body.extract_joint_kinetics(cgm1_gait)
        openmaLib.renameOpenMAtoVicon(kinetics)
        
        
        # append new parameters to the gait trial    
        trialTools.addTimeSequencesToTrial(dynamicTrial,kinematics)
        trialTools.addTimeSequencesToTrial(dynamicTrial,kinetics)

        # add property
        dynamicTrial.setProperty("MODEL:NAME","CGM1")
        dynamicTrial.setProperty('MODELLING:PROCESSING',"openMA")

        # ----- WRITER --------
        if ma.io.write(dynamicNode,str(DATA_PATH + reconstructFilenameLabelled[:-4] + "_openmaCGM1.c3d")):
            logging.info( "File ( %s) reconstructed ( suffix : _openmaCGM1.c3d) " % (reconstructFilenameLabelled) )                 
   
    else: 
        logging.error("Nexus Not Connected")        
        
        