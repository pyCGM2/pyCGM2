# -*- coding: utf-8 -*-
import numpy as np
import pdb
import logging
import json

import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.DEBUG)

# btk
pyCGM2.CONFIG.addBtk()
import btk

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm, modelFilters, modelDecorator, frame
import pyCGM2.enums as pyCGM2Enums
from collections import OrderedDict


class translator_tests(): 

    @classmethod
    def cgm2_3(cls):    


        CONTENT_INPUTS_CGM2_3 ="""
            {  
              "Translators" : {
                    "LASI":"",
                    "RASI":"",
                    "LPSI":"",
                    "RPSI":"",
                    "RTHI":"RTHL",
                    "RKNE":"",
                    "RTHIAP":"RTHAP",
                    "RTHIAD":"RTHAD",
                    "RTIB":"RTIBL",
                    "RANK":"RANK",
                    "RTIBAP":"RTIAP",
                    "RTIBAD":"RTIAD",
                    "RHEE":"",
                    "RTOE":"",
                    "LTHI":"LTHL",
                    "LKNE":"",
                    "LTHIAP":"LTHAP",
                    "LTHIAD":"LTHAD",
                    "LTIB":"LTIBL",
                    "LANK":"",
                    "LTIBAP":"LTIAP",
                    "LTIBAD":"LTIAD",
                    "LHEE":"",
                    "LTOE":""
              }  
            }
            """
        inputs = json.loads(CONTENT_INPUTS_CGM2_3,object_pairs_hook=OrderedDict)
        translators = inputs["Translators"]

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.3\\"
        staticFilename = "static.c3d"
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))


        acqStatic2 =  btkTools.applyTranslators(acqStatic,translators)
        
        
        
        btkTools.smartWriter(acqStatic2,"test.c3d") 

        
        
if __name__ == "__main__":

    translator_tests.cgm2_3()   
