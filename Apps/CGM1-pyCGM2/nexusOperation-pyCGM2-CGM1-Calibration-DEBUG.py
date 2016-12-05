# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:41:34 2016

@author: Fabien Leboeuf
"""
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import json
import pdb


# pyCGM2 settings
import pyCGM2
pyCGM2.pyCGM2_CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
pyCGM2.pyCGM2_CONFIG.addNexusPythonSdk()
import ViconNexus

# openMA
pyCGM2.pyCGM2_CONFIG.addOpenma()
import ma.io
import ma.body


# pyCGM2 libraries 
from pyCGM2.Core.Model.CGM2 import cgm, modelFilters, modelDecorator
from pyCGM2.Core.Tools import btkTools
import pyCGM2.Core.enums as pyCGM2Enums 
#from pyCGM2.Core.Report import plot
from pyCGM2 import  smartFunctions 
from pyCGM2.Core.Report import plot



if __name__ == "__main__":
    
        
    plt.close("all")
    #pyNEXUS = ViconNexus.ViconNexus()    
    #NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()

    NEXUS_PYTHON_CONNECTED = True   
     
    if NEXUS_PYTHON_CONNECTED: # run Operation

        #---- INPUTS------     
        flag_leftFlatFoot =  True
        flag_rightFlatFoot =  True
        markerDiameter = 14
        
        
        #---- DATA ------- 
        DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\CGM1-NexusPlugin\\pyCGM2-CGM1-basic\\"
    
        calibrateFilenameLabelledNoExt = "static Cal 01"
        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"
        reconstructFilenameLabelled = calibrateFilenameLabelled   
        
        logging.info( "data Path: "+ DATA_PATH )   
        logging.info( "calibration file: "+ calibrateFilenameLabelled)
        logging.info( "reconstruction file: "+ reconstructFilenameLabelled ) 
        
        # subject mp
        mp={
        'mass'   : 71.0,                
        'leftLegLength' : 860.0,
        'rightLegLength' : 865.0 ,
        'leftKneeWidth' : 102.0,
        'rightKneeWidth' : 103.4,
        'leftAnkleWidth' : 75.3,
        'rightAnkleWidth' : 72.9,       
        }

 
        # -----------CGM STATIC CALIBRATION--------------------
        model=cgm.CGM1ModelInf()
        model.configure()
        model.addAnthropoInputParameter(mp)


        # reader
        
        acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))

        # initial static filter
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                                            markerDiameter=markerDiameter).compute() 

        
#        # cgm decorator
#        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=25, displayMarkers = True)
#        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=25, side="both")
#        
#        # final static filter
#        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, 
#                                   useLeftKJCnode="LKJC_kad", useLeftAJCnode="LAJC_mid", 
#                                   useRightKJCnode="RKJC_kad", useRightAJCnode="RAJC_mid",
#                                   useLeftTibialTorsion = True,useRightTibialTorsion = True,
#                                   markerDiameter=25).compute()

        # -----------CGM RECONSTRUCTION--------------------
        acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native,
                                                  markerDiameter=markerDiameter)

        modMotion.compute() 


        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="")
#
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqGait,"SACR","midASIS","LPSI")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                               segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix="")        


       
        # writer
        btkTools.smartWriter(acqGait,str(DATA_PATH + reconstructFilenameLabelled[:-4] + "_pyCGM2_CGM1.c3d"))
        logging.info( "[pyCGM2] : file ( %s) reconstructed " % (reconstructFilenameLabelled))
        
        # static angle profile
        model= None 
        subject=None       
        experimental=None
        smartFunctions.staticProcessing_cgm1(str(reconstructFilenameLabelled[:-4] + "_pyCGM2_CGM1.c3d"), DATA_PATH,
                                             model,  subject, experimental)
        
        
        # reader    
        kinematicFileNode = ma.io.read(str(DATA_PATH + reconstructFilenameLabelled[:-4] + "_pyCGM2_CGM1.c3d"))
        kinematicTrial = kinematicFileNode.findChild(ma.T_Trial)
            
        plot.gaitKinematicsTemporalPlotPanel([kinematicTrial],["Vicon"])
                 
    else: 
        logging.error("Nexus Not Connected")  
    