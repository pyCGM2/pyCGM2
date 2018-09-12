# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:05:09 2016

@author: fabien Leboeuf( salford Univ)
"""

import sys
import pdb
import logging


# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)


# vicon
pyCGM2.addNexusPythonSdk()
import ViconNexus

# openMA
pyCGM2.addOpenma()
import ma.io
import ma.body
   
# pyCGM2 libraries   
from  pyCGM2.Tools  import trialTools
from pyCGM2 import  smartFunctions    
   
if __name__ == "__main__":

    #pyNEXUS = ViconNexus.ViconNexus()    
    #NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()      

    NEXUS_PYTHON_CONNECTED = True

    if NEXUS_PYTHON_CONNECTED:       
    
        Calibration = False
        if Calibration:
            calibrateFilenameLabelledNoExt = None   
        else:
            calibrateFilenameLabelledNoExt = "static Cal 01"
            
        flag_leftFlatFoot =  True
        flag_rightFlatFoot =  True
        markerDiameter = 14
        pointSuffix = ""  

        enableProcessing = True
        normativeDataInput = "Schwartz2008_Free"
    
        #---- DATA ------ 
        DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\CGM1-NexusPlugin\\openMA-CGM1-basic\\"
        reconstructFilenameLabelledNoExt ="gait Trial 01"  
        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
        
        
        if calibrateFilenameLabelledNoExt is None:
            logging.warning("Static Processing")
            staticProcessing = True
            calibrateFilenameLabelled = "static Cal 01" + ".c3d" 
            reconstructFilenameLabelledNoExt = "static Cal 01"  
            reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
        else:
            staticProcessing = False
            reconstructFilenameLabelledNoExt = "gait Trial 01"  
            reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
            calibrateFilenameLabelled = reconstructFilenameLabelled        
        
    
        
        logging.info( "data Path: "+ DATA_PATH )   
        logging.info( "calibration file: "+ calibrateFilenameLabelled)
        logging.info( "reconstruction file: "+ reconstructFilenameLabelled )     
            
        # ----- SUBJECT --------
        sub = ma.Subject(str("Subject Name")) 
        sub.setProperty("mass",ma.Any(71.0))
        sub.setProperty("height",ma.Any(1756.0))
        sub.setProperty("markerDiameter",ma.Any(markerDiameter))
        sub.setProperty("leftLegLength",ma.Any(860))
        sub.setProperty("leftKneeWidth",ma.Any(102.0))
        sub.setProperty("leftAnkleWidth",ma.Any(73.4))
        sub.setProperty("rightLegLength",ma.Any(865.0))
        sub.setProperty("rightKneeWidth",ma.Any(103.4))
        sub.setProperty("rightAnkleWidth",ma.Any(72.9))
        
        # ----- STATIC CALIBRATION --------
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