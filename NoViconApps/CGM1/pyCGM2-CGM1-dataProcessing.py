# -*- coding: utf-8 -*-

import os
import logging
import matplotlib.pyplot as plt
import json
import pdb
import cPickle
import json
from collections import OrderedDict
from shutil import copyfile
import argparse


# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# openMA
#import ma.io
#import ma.body

#btk
import btk


# pyCGM2 libraries    
from pyCGM2 import  smartFunctions 


#

import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Gait Processing')
    parser.add_argument('--pointSuffix', type=str, help='force suffix')
    args = parser.parse_args()  
    
    
    
    
    # --------------------pyCGM2 SETTINGS FILES ------------------------------


    # info file
    infoSettings = json.loads(open('pyCGM2.info').read(),object_pairs_hook=OrderedDict)
        
    if args.pointSuffix is not None:
        pointSuffix = args.pointSuffix
    else:
        pointSuffix = infoSettings["Processing"]["Point suffix"]        
        
    normativeData = infoSettings["Processing"]["Normative data"]


    # -----infos--------     
    model = None if  infoSettings["Modelling"]["Model"]=={} else infoSettings["Modelling"]["Model"]  
    subject = None if infoSettings["Processing"]["Subject"]=={} else infoSettings["Processing"]["Subject"] 
    experimental = None if infoSettings["Processing"]["Experimental conditions"]=={} else infoSettings["Processing"]["Experimental conditions"] 

    # --------------------------PROCESSING --------------------------------

    DATA_PATH = infoSettings["Modelling"]["Trials"]["DataPath"]
    motionTrialFilenames = infoSettings["Modelling"]["Trials"]["Motion"]
    
    smartFunctions.gaitProcessing_cgm1 (motionTrialFilenames, DATA_PATH,
                           model,  subject, experimental,
                           pointLabelSuffix = pointSuffix,
                           plotFlag= True, 
                           exportBasicSpreadSheetFlag = False,
                           exportAdvancedSpreadSheetFlag = False,
                           exportAnalysisC3dFlag = False,
                           consistencyOnly = True,
                           normativeDataDict = normativeData)


   