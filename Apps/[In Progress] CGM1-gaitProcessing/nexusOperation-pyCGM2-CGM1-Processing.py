# -*- coding: utf-8 -*-
import sys
import pdb
import logging
import matplotlib.pyplot as plt
import argparse

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
pyCGM2.CONFIG.addNexusPythonSdk()
import ViconNexus


# openMA
pyCGM2.CONFIG.addOpenma()
import ma.io
import ma.body
    
# pyCGM2 libraries    
from pyCGM2 import  smartFunctions 
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CGM')
    parser.add_argument('--StaticProcessing', action='store_true', help='static processing' )
    parser.add_argument('-p','--plot', action='store_true', help='plot results' )
    parser.add_argument('-x','--xls', action='store_true', help='return spreadsheet' )
    parser.add_argument('--pointSuffix', default="", type=str)
    parser.add_argument('--author', default="Schwartz2008", type=str)
    parser.add_argument('--modality', default="Free", type=str)    
    
    args = parser.parse_args()
    
    logging.info(args)

   
    plt.close("all")    
    


    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected() 

       
        
    if NEXUS_PYTHON_CONNECTED: 
        
        
        #---- INPUTS -----
        staticProcessing = args.StaticProcessing 
        plotFlag = args.plot
        exportSpreadSheetFlag = args.xls
        normativeDataInput = str(args.author+"_"+ args.modality) #"Schwartz2008_VeryFast"       
        if  args.pointSuffix == "":
            pointSuffix = ""
        else:
            pointSuffix = args.pointSuffix
                    
        # ----DATA-----        
        DATA_PATH, reconstructedFilenameLabelledNoExt = pyNEXUS.GetTrialName() 
        reconstructedFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"        

        print "data Path: "+ DATA_PATH    
        print "reconstructed file: "+ reconstructedFilenameLabelled

        # -----INFOS--------     
        model=None  
        subject=None
        experimental=None

        # ----PROCESSING-----
        if staticProcessing:
            # static angle profile
            smartFunctions.staticProcessing_cgm1(reconstructedFilenameLabelled, DATA_PATH,
                                                 model,  subject, experimental,
                                                 pointLabelSuffix = pointSuffix)
        else:
            normativeData = { "Author": normativeDataInput[:normativeDataInput.find("_")],"Modality": normativeDataInput[normativeDataInput.find("_")+1:]}
            smartFunctions.gaitProcessing_cgm1 (reconstructedFilenameLabelled, DATA_PATH,
                                   model,  subject, experimental,
                                   pointLabelSuffix = pointSuffix,
                                   plotFlag= plotFlag, 
                                   exportBasicSpreadSheetFlag = exportSpreadSheetFlag,
                                   exportAdvancedSpreadSheetFlag = exportSpreadSheetFlag,
                                   exportAnalysisC3dFlag = False,
                                   consistencyOnly = True,
                                   normativeDataDict = normativeData)

    else: 
        logging.error("[pyCGM2] : Nexus Not Connected")


