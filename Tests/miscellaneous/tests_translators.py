# -*- coding: utf-8 -*-
import numpy as np
import pdb
import logging
import json

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)


# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model import modelFilters, modelDecorator, frame
import pyCGM2.enums as pyCGM2Enums
from collections import OrderedDict
from pyCGM2.Utils import files

class translator_tests():

    @classmethod
    def cgm2_3_jsonContent(cls):


        CONTENT_INPUTS_CGM2_3 ="""
            {
              "Translators" : {
                    "LASI":"None",
                    "RASI":"None",
                    "LPSI":"None",
                    "RPSI":"None",
                    "RTHI":"RTHL",
                    "RKNE":"None",
                    "RTHAP":"RTHAP",
                    "RTHAD":"RTHAD",
                    "RTIB":"RTIBL",
                    "RANK":"RANK",
                    "RTIAP":"RTIAP",
                    "RTIAD":"RTIAD",
                    "RHEE":"None",
                    "RTOE":"None",
                    "LTHI":"LTHL",
                    "LKNE":"None",
                    "LTHAP":"LTHAP",
                    "LTHAD":"LTHAD",
                    "LTIB":"LTIBL",
                    "LANK":"None",
                    "LTIAP":"LTIAP",
                    "LTIAD":"LTIAD",
                    "LHEE":"None",
                    "LTOE":"None"
              }
            }
            """
        inputs = json.loads(CONTENT_INPUTS_CGM2_3,object_pairs_hook=OrderedDict)
        translators = inputs["Translators"]

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\Translators\\cgm2.3\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))


        acqStatic2 =  btkTools.applyTranslators(acqStatic,translators)


        btkTools.smartWriter(acqStatic2,"test.c3d")

    @classmethod
    def cgm2_3_yamlFile(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "operations\\Translators\\cgm2.3\\"
        staticFilename = "static.c3d"

        translators = files.getTranslators(MAIN_PATH, translatorType = "CGM2_3.translators")

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        acqStatic2 =  btkTools.applyTranslators(acqStatic,translators)


        btkTools.smartWriter(acqStatic2,"test_yamlfile.c3d")


if __name__ == "__main__":

    #translator_tests.cgm2_3_jsonContent()
    translator_tests.cgm2_3_yamlFile()
