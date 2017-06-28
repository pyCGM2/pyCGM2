# -*- coding: utf-8 -*-

import pyCGM2
import logging

import json
import os
from shutil import copyfile
from collections import OrderedDict

def manage_pycgm2SessionInfos(DATA_PATH,subject):
    
    if not os.path.isfile( DATA_PATH + subject+"-pyCGM2.info"):
        copyfile(str(pyCGM2.CONFIG.PYCGM2_SESSION_SETTINGS_FOLDER+"pyCGM2.info"), str(DATA_PATH + subject+"-pyCGM2.info"))
        logging.warning("Copy of pyCGM2.info from pyCGM2 Settings folder")
        infoSettings = json.loads(open(DATA_PATH +subject+'-pyCGM2.info').read(),object_pairs_hook=OrderedDict)
    else:
        infoSettings = json.loads(open(DATA_PATH +subject+'-pyCGM2.info').read(),object_pairs_hook=OrderedDict)

    return infoSettings

def manage_pycgm2Translators(DATA_PATH, translatorType = "CGM1.translators"):
    #  translators management 
    if os.path.isfile( DATA_PATH + translatorType):
       logging.warning("local translator found")
       sessionTranslators = json.loads(open(DATA_PATH + translatorType).read(),object_pairs_hook=OrderedDict)
       translators = sessionTranslators["Translators"]
       return translators
    else:
       return False

