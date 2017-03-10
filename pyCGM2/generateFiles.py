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


def generateCGM1_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM1-pyCGM2.inputs"):    
        inputs = json.loads(CONTENT_INPUTS_CGM1,object_pairs_hook=OrderedDict)
        
        F = open(str(userAppData_path+"CGM1-pyCGM2.inputs"),"w") 
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()