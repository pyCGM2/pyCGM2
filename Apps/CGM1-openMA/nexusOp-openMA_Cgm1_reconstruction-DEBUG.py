# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:05:09 2016

@author: fabien Leboeuf( salford Univ)
"""

import sys
import pdb
import logging


try:
    import pyCGM2.pyCGM2_CONFIG 
except ImportError:
    logging.error("[pyCGM2] : pyCGM2 module not in your python path")

# vicon
pyCGM2.pyCGM2_CONFIG.addNexusPythonSdk()
import ViconNexus

# openMA
pyCGM2.pyCGM2_CONFIG.addOpenma()
import ma.io
import ma.body

# pyCGM2
import pyCGM2.Core.Model.openmaLib as openmaLib
import pyCGM2.Core.Tools.trialTools as CGM2trialTools
   
if __name__ == "__main__":

    #pyNEXUS = ViconNexus.ViconNexus()    
    #NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()      
    
    flag_leftFlatFoot =  True
    flag_rightFlatFoot =  True
    calibrateFilenameLabelled = "static Cal 01.c3d"

    #---- DATA ------ 
    DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\CGM1-NexusPlugin\\test0\\"
    reconstructFilenameLabelledNoExt ="gait Trial 01"  
    reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
    
    print "data Path: "+ DATA_PATH    
    print "calibration file: "+ calibrateFilenameLabelled
    print "reconstruction file: "+ reconstructFilenameLabelled


    
     # ----- SUBJECT --------
    sub = ma.Subject(str("Fabien")) 
    sub.setProperty("mass",ma.Any(71.0))
    sub.setProperty("height",ma.Any(1756.0))
    sub.setProperty("markerDiameter",ma.Any(14.0))
    sub.setProperty("leftLegLength",ma.Any(860))
    sub.setProperty("leftKneeWidth",ma.Any(102.0))
    sub.setProperty("leftAnkleWidth",ma.Any(73.4))
    sub.setProperty("rightLegLength",ma.Any(865.0))
    sub.setProperty("rightKneeWidth",ma.Any(103.4))
    sub.setProperty("rightAnkleWidth",ma.Any(72.9))
    
    # ----- STATIC CALIBRATION --------
    staticNode = ma.io.read(str(DATA_PATH+calibrateFilenameLabelled))

    # calibration
    cgm1_static = ma.body.PluginGait(ma.body.Region_Lower, ma.body.Side_Both)
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
    openmaLib.renameOpenMAtoVicon(kinematics)
    kinetics = ma.body.extract_joint_kinetics(cgm1_gait)
    openmaLib.renameOpenMAtoVicon(kinetics)
    

    # append new parameters to the gait trial    
    CGM2trialTools.addTimeSequencesToTrial(dynamicTrial,kinematics)
    CGM2trialTools.addTimeSequencesToTrial(dynamicTrial,kinetics)

    # add property
    dynamicTrial.setProperty('MODELLING:NAME',"CGM1")
    
    # ----- WRITER --------
    if ma.io.write(dynamicNode,str(DATA_PATH + reconstructFilenameLabelled[:-4] + "_CGM1.c3d")):
        print "[pyCGM2] : file ( %s) reconstructed " % (reconstructFilenameLabelled)                 
