# -*- coding: utf-8 -*-
"""
 
Usage:
    file.py
    file.py -h | --help
    file.py --version
    file.py  Calibration [-lr] [--markerDiameter=<n>]  
    file.py  Calibration [-lr] [--markerDiameter=<n> --pointSuffix=<ps>] 
    file.py  <staticFile> [-lr] [--markerDiameter=<n>]
    file.py  <staticFile> [-lr] [--markerDiameter=<n>] [-p ]
    file.py  <staticFile> [-lr] [--markerDiameter=<n>] [-p  --author=<authorYear> --modality=<modalitfy>]
    file.py  <staticFile> [-lr] [--markerDiameter=<n> --pointSuffix=<ps>]         
    file.py  <staticFile> [-lr] [--markerDiameter=<n> --pointSuffix=<ps>] [-p | --plot --author=<authorYear> --modality=<modalitfy>]    
    
 
Arguments:

 
Options:
    -h --help   Show help message
    -l          Enable left flat foot option
    -r          Enable right flat foot option
    -p   Enable gait Plots  
    --markerDiameter=<n>  marker diameter [default: 14].
    --pointSuffix=<ps>  suffix associated with classic vicon output label  [default: ""].
    --author=<authorYear>   Name and year of the Normative Data base used [default: Schwartz2008]
    --modality=<modalitfy>  Modality of the Normative Database used  [default: Free]

"""

import sys
import pdb
import logging
from docopt import docopt


# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# vicon
pyCGM2.CONFIG.addNexusPythonSdk()
import ViconNexus

# openMA
pyCGM2.CONFIG.addOpenma()
import ma.io
import ma.body
   
# pyCGM2 libraries   
import pyCGM2.Model.openmaLib as openmaLib
from  pyCGM2.Tools  import trialTools   
   


if __name__ == "__main__":
    args = docopt(__doc__, version='0.1')    
    
    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected() 

        
    if NEXUS_PYTHON_CONNECTED: # run operation
        
        #---- INPUTS------
        if args['Calibration']:
            calibrateFilenameLabelledNoExt = None  #sys.argv[1] 
        else:
            calibrateFilenameLabelledNoExt = args['<staticFile>']  #sys.argv[1] 

        flag_leftFlatFoot =  args['-l'] #bool(int(sys.argv[2]))
        flag_rightFlatFoot = args['-r'] #bool(int(sys.argv[3]))
        markerDiameter =  float(args['--markerDiameter']) #float(sys.argv[4])
       
        
        
        
   
        
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
        openmaLib.renameOpenMAtoVicon(kinematics)
        # append new parameters to the gait trial    
        trialTools.addTimeSequencesToTrial(dynamicTrial,kinematics)

        if not staticProcessing:
            kinetics = ma.body.extract_joint_kinetics(cgm1_gait)
            openmaLib.renameOpenMAtoVicon(kinetics)

            trialTools.addTimeSequencesToTrial(dynamicTrial,kinetics)
    
        # add property
        dynamicTrial.setProperty('MODEL:NAME',"CGM1")
        dynamicTrial.setProperty('MODEL:PROCESSOR',"openMA")
        
        # ----- WRITER --------
        if ma.io.write(dynamicNode,str(DATA_PATH + reconstructFilenameLabelled[:-4] + "_openmaCGM1.c3d")):
            logging.info( "file ( %s) reconstructed ( suffix : _openmaCGM1.c3d) " % (reconstructFilenameLabelled) )                 


              
    else: 
        logging.error(" Nexus Not Connected")        
