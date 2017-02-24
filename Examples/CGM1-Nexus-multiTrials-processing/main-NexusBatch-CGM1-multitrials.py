# -*- coding: utf-8 -*-
import logging
import json
import pdb

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model.CGM2 import cgm, modelFilters, modelDecorator, forceplates,bodySegmentParameters
from pyCGM2 import  smartFunctions 


def checkCGM1_StaticMarkerConfig(acqStatic):

    out = dict()

    # medial ankle markers
    out["leftMedialAnkleFlag"] = True if btkTools.isPointsExist(acqStatic,["LMED","LANK"]) else False
    out["rightMedialAnkleFlag"] = True if btkTools.isPointsExist(acqStatic,["RMED","RANK"]) else False

    # medial knee markers
    out["leftMedialKneeFlag"] = True if btkTools.isPointsExist(acqStatic,["LMEPI","LKNE"]) else False
    out["rightMedialKneeFlag"] = True if btkTools.isPointsExist(acqStatic,["RMEPI","RKNE"]) else False


    # kad
    out["leftKadFlag"] = True if btkTools.isPointsExist(acqStatic,["LKAX","LKD1","LKD2"]) else False
    out["rightKadFlag"] = True if btkTools.isPointsExist(acqStatic,["RKAX","RKD1","RKD2"]) else False

    return out



if __name__ == "__main__":

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED: # run Operation

        DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\PIG standard\\basic\\"

        # ---- load pyCGM2 inputs    
        INPUTS = json.loads(open(str(DATA_PATH +'pyCGM2.inputs')).read())
        pointSuffix = INPUTS["Calibration"]["Point suffix"]
        
        # --------------------------DATA--------------------------------------
        
        staticTrialName = INPUTS["Static trial"]
        dynamicTrialNames = INPUTS["Dynamic trials"]
        
        # --------------------------STATIC CALBRATION--------------------------                
        staticTrialNameNoExt = staticTrialName[:-4]
        
        NEXUS.OpenTrial( str(DATA_PATH+staticTrialNameNoExt), 30 )
        NEXUS.RunPipeline( 'pyCGM2-CGM1-Calibration', 'Private', 45 )
        NEXUS.SaveTrial( 30 )

   
        # --------------------------FITTING--------------------------
        for dynamicTrialName in dynamicTrialNames:
            dynamicTrialNameNoExt = dynamicTrialName[:-4]
            NEXUS.OpenTrial( str(DATA_PATH+dynamicTrialNameNoExt), 30 )
            NEXUS.RunPipeline( 'pyCGM2-CGM1-Fitting', 'Private', 45 )
            NEXUS.SaveTrial( 30 )
           
                        
    # --------------------------Multi Modelled trial PROCESSING --------------------------------

    # -----infos-------- 
    model = None if  INPUTS["Model"]=={} else INPUTS["Model"]  
    subject = None if INPUTS["Subject"]=={} else INPUTS["Subject"] 
    experimental = None if INPUTS["Experimental conditions"]=={} else INPUTS["Experimental conditions"] 

    normativeData = INPUTS["Normative data"]

    # -----processing pipeline-------- 
    modelledDynamicTrials = dynamicTrialNames
    
    smartFunctions.gaitProcessing_cgm1 (modelledDynamicTrials, DATA_PATH,
                           model,  subject, experimental,
                           pointLabelSuffix = pointSuffix,
                           plotFlag= True, 
                           exportBasicSpreadSheetFlag = False,
                           exportAdvancedSpreadSheetFlag = False,
                           exportAnalysisC3dFlag = False,
                           consistencyOnly = True,
                           normativeDataDict = normativeData)