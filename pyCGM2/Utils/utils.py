# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Utils
#APIDOC["Draft"]=False
#--end--

import pyCGM2
import pyCGM2; LOGGER = pyCGM2.LOGGER

def toBool(text):
    """convert text to bool

    Args:
        text (str): text

    """
    return True if text == "True" else False

def isInRange(val, min, max):
    """check if value is in range

    Args:
        val (double): value
        min (double): minimim value
        max (double): maximum value

    """

    if val<min or val>max:
        return False
    else:
        return True

def str(unicodeVariable):
    return unicodeVariable.encode(pyCGM2.ENCODER)

def checkSimilarElement(listData):
    if(len(set(listData))==1):
        for it in set(listData):
            return it
        return True
    else:
        LOGGER.logger.error("[pyCGM2] items are different in the inputed list" )
        return False

def getSimilarElement(listData):
    out = list()
    for it in set(listData):
        out = it
    return out
