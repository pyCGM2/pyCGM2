# -*- coding: utf-8 -*-
import pyCGM2
import pyCGM2; LOGGER = pyCGM2.LOGGER

def toBool(text):
    return True if text == "True" else False

def isInRange(val, min, max):

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
