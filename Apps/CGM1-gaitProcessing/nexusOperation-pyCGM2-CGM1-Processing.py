# -*- coding: utf-8 -*-
"""
 
Usage:
    file.py
    file.py -h | --help
    file.py --version
    file.py StaticAngleProfile
    file.py [--plot --c3d --xls --pointSuffix=<ps>]    
    file.py [--plot --c3d --xls --pointSuffix=<ps> --author=<authorYear> --modality=<modalitfy>]  

 
Arguments:

 
Options:
    -h --help   Show help message
    --c3d
    --plot
    --xls
    --pointSuffix=<ps>  suffix associated with classic vicon output label  [default: ""].
    --author=<authorYear>   Name and year of the Normative Data base used [default: Schwartz2008]
    --modality=<modalitfy>  Modality of the Normative Database used  [default: Free]
"""

import sys
import pdb
import logging
import matplotlib.pyplot as plt
from docopt import docopt

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
from pyCGM2 import  smartFunctions 
    
if __name__ == "__main__":
    plt.close("all")
    args = docopt(__doc__, version='0.1')
    plt.close("all")    
    


    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected() 

       
        
    if NEXUS_PYTHON_CONNECTED: 
        
        
        #---- INPUTS -----
        staticProcessing = args['StaticAngleProfile']
        plotFlag = args['--plot'] #bool(int(sys.argv[1]))
        exportSpreadSheetFlag = args['--xls'] #bool(int(sys.argv[2]))
        exportAnalysisC3dFlag = args['--c3d'] #bool(int(sys.argv[3]))
        normativeDataInput = str(args['--author']+"_"+ args['--modality']) #sys.argv[4] #"Schwartz2008_VeryFast"
        if  args['--pointSuffix'] == '""':
            pointSuffix = ""
        else:
            pointSuffix = args['--pointSuffix']# sys.argv[5]         

            
        print args
        
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
                                   exportAnalysisC3dFlag = exportAnalysisC3dFlag,
                                   consistencyOnly = True,
                                   normativeDataDict = normativeData)

    else: 
        logging.error("[pyCGM2] : Nexus Not Connected")


