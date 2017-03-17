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


# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus
    
# pyCGM2 libraries    
from pyCGM2 import  smartFunctions 
from pyCGM2.Tools import btkTools,nexusTools
    
if __name__ == "__main__":
   
    plt.close("all")
    DEBUG = False

    parser = argparse.ArgumentParser(description='CGM1 Calibration')
    parser.add_argument('--pointSuffix', type=str, help='force suffix')
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------INPUTS ------------------------------------

        if DEBUG:
            DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\CGM1-NexusPlugin\\New Session 3\\"
            reconstructedFilenameLabelledNoExt = "MRI-US-01, 2008-08-08, 3DGA 12"
            NEXUS.OpenTrial( str(DATA_PATH+reconstructedFilenameLabelledNoExt), 30 )


        else:
            DATA_PATH, reconstructedFilenameLabelledNoExt = NEXUS.GetTrialName()


        reconstructedFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ reconstructedFilenameLabelled)

        # ----- Subject -----
        # need subject to find input files 
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.ckeckActivatedSubject(NEXUS,subjects,"LASI")
        logging.info(  "Subject name : " + subject  )

        # ---- pyCGM2 input files ----

       # info file
        if not os.path.isfile( DATA_PATH + subject+"-pyCGM2.info"):
            copyfile(str(pyCGM2.CONFIG.PYCGM2_SETTINGS_FOLDER+"pyCGM2.info"), str(DATA_PATH + subject+"-pyCGM2.info"))
            logging.warning("Copy of pyCGM2.info from pyCGM2 Settings folder")
            infoSettings = json.loads(open(DATA_PATH +subject+'-pyCGM2.info').read(),object_pairs_hook=OrderedDict)
        else:
            infoSettings = json.loads(open(DATA_PATH +subject+'-pyCGM2.info').read(),object_pairs_hook=OrderedDict)

        # ---- configuration parameters ----
        if args.pointSuffix is not None:
            pointSuffix = args.pointSuffix
        else:
            pointSuffix = infoSettings["Processing"]["Point suffix"]        
        
        normativeData = infoSettings["Processing"]["Normative data"]


        # -----infos--------     
        model = None if  infoSettings["Processing"]["Model"]=={} else infoSettings["Processing"]["Model"]  
        subject = None if infoSettings["Processing"]["Subject"]=={} else infoSettings["Processing"]["Subject"] 
        experimental = None if infoSettings["Processing"]["Experimental conditions"]=={} else infoSettings["Processing"]["Experimental conditions"] 

        # --------------------------PROCESSING --------------------------------
        # pycgm2-filter pipeline are gathered in a single function
        
        smartFunctions.gaitProcessing_cgm1 (reconstructedFilenameLabelled, DATA_PATH,
                               model,  subject, experimental,
                               pointLabelSuffix = pointSuffix,
                               plotFlag= True, 
                               exportBasicSpreadSheetFlag = False,
                               exportAdvancedSpreadSheetFlag = False,
                               exportAnalysisC3dFlag = False,
                               consistencyOnly = True,
                               normativeDataDict = normativeData)

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")


