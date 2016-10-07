# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 10:40:17 2016

@author: fabien Leboeuf ( Salfordd Univ)
"""
import sys
import pdb
import logging
import matplotlib.pyplot as plt

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

import pyCGM2.smartFunctions as CGM2smart
    
if __name__ == "__main__":
    plt.close("all")    
    


    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected() 

    NEXUS_PYTHON_CONNECTED = True   
        
    if NEXUS_PYTHON_CONNECTED: 
        
        
        #---- INPUTS ----- 
        plotFlag = bool(int(sys.argv[1]))
        exportSpreadSheetFlag = bool(int(sys.argv[2]))
        exportAnalysisC3dFlag = bool(int(sys.argv[3]))
        normativeDataInput = sys.argv[3] #"Schwartz2008_VeryFast"
        
        normativeData = { "Author": normativeDataInput[:normativeDataInput.find("_")],"Modality": normativeDataInput[normativeDataInput.find("_")+1:]} 
        
        # ----DATA-----        
        DATA_PATH, reconstructedFilenameLabelledNoExt = pyNEXUS.GetTrialName() 
        reconstructedFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"        

        print "data Path: "+ DATA_PATH    
        print "reconstructed file: "+ reconstructedFilenameLabelled

        # -----INFOS--------     
        model={"HJC": "har",
           "KinematicsFitting": "OpenSim"}    
    
        subject={"ipp": "000",
                 "firstname": "NA",
                 "surname": "NA",
                 "sex": "M",
                 "dob": "-"
                         }
                         
        experimental={            
                    "date": "NA",
                    "doctor": "NA",
                    "context": "Research",
                    "task": "S",
                    "shoe": "N",
                    "orthosis Prothesis": "N",
                    "external Device": "N",
                    "person assistance": "N"
                     }

        # ----PROCESSING-----
        CGM2smart.gaitProcessing_cgm1 (reconstructedFilenameLabelled, DATA_PATH,
                               model,  subject, experimental, 
                               plotFlag= plotFlag, 
                               exportSpreadSheetFlag = True,
                               exportAnalysisC3dFlag = True,
                               normativeDataDict = normativeData)

    else: 
        logging.error("[pyCGM2] : Nexus Not Connected")


