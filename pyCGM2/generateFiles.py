# -*- coding: utf-8 -*-
import os
import json
from collections import OrderedDict

CONTENT_INPUTS_CGM1 ="""
    {  
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Calibration" : {
        "Left flat foot" : 1 ,
        "Right flat foot" : 1 
      },
      "Fitting" : {
        "Moment Projection" : "Distal"
      }   
    }
    """


CONTENT_INPUTS_CGM1_1 ="""
    {  
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Calibration" : {
        "Left flat foot" : 1 ,
        "Right flat foot" : 1 
      },
      "Fitting" : {
        "Moment Projection" : "Proximal"
      }   
    }
    """

CONTENT_INPUTS_CGM2_1 ="""
    {  
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Calibration" : {
        "HJC regression" : "Hara",
        "Left flat foot" : 1 ,
        "Right flat foot" : 1 
      },
      "Fitting" : {
        "Moment Projection" : "Proximal"
      }   
    }
    """

CONTENT_INPUTS_CGM2_2 ="""
    {  
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Calibration" : {
        "HJC regression" : "Hara",
        "Left flat foot" : 1 ,
        "Right flat foot" : 1 
      },
      "Fitting" : {
        "Moment Projection" : "Proximal",
        "Weight" :{
            "LASI" : 100,
            "RASI" : 100,
            "LPSI" : 100,
            "RPSI" : 100,
            "LTHI" : 100,
            "LKNE" : 100,
            "LTIB":  100,
            "LANK" : 100,
            "LHEE" : 100,
            "LTOE" : 100,            
            "RTHI" : 100,
            "RKNE" : 100,
            "RTIB":  100,
            "RANK" : 100,
            "RHEE" : 100,
            "RTOE" : 100
        }
      }   
    }
    """





def generateCGM1_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM1-pyCGM2.inputs"):    
        inputs = json.loads(CONTENT_INPUTS_CGM1,object_pairs_hook=OrderedDict)
        
        F = open(str(userAppData_path+"CGM1-pyCGM2.inputs"),"w") 
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()
        
def generateCGM1_1_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM1_1-pyCGM2.inputs"):    
        inputs = json.loads(CONTENT_INPUTS_CGM1_1,object_pairs_hook=OrderedDict)
        
        F = open(str(userAppData_path+"CGM1_1-pyCGM2.inputs"),"w") 
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()

def generateCGM2_1_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM2_1-pyCGM2.inputs"):    
        inputs = json.loads(CONTENT_INPUTS_CGM2_1,object_pairs_hook=OrderedDict)
        
        F = open(str(userAppData_path+"CGM2_1-pyCGM2.inputs"),"w") 
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()        
        
def generateCGM2_2_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM2_2-pyCGM2.inputs"):    
        inputs = json.loads(CONTENT_INPUTS_CGM2_2,object_pairs_hook=OrderedDict)
        
        F = open(str(userAppData_path+"CGM2_2-pyCGM2.inputs"),"w") 
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()         