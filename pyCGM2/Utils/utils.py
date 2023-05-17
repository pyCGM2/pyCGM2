# -*- coding: utf-8 -*-

import pyCGM2
import pyCGM2; LOGGER = pyCGM2.LOGGER

class FrameConverter():
    def __init__(self,firstFrame,lastFrame, nppf):
        self.firstFrame = firstFrame
        self.lastFrame = lastFrame
        self.nppf = nppf

    def toAnalog(self,pointFrame,reframeToOne=False):
        if not reframeToOne:
           val = (pointFrame-self.firstFrame)*self.nppf + self.firstFrame
        else: 
           val = (pointFrame-self.firstFrame)*self.nppf
        
        return int(val)

    def toPoint(self,analogFrame,reframeToOne=False):
        if not reframeToOne:
            val =  (analogFrame-self.firstFrame)/self.nppf + self.firstFrame
        else:
            val = (analogFrame-self.firstFrame)/self.nppf 
        
        return int(val)


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


def homogeneizeArguments(argv,kwargs):
    for arg in argv:
        if isinstance(arg,dict):
            for argvKey in arg.keys():
                if argvKey in kwargs.keys():
                    LOGGER.logger.warning("The positional argument (%s) is already defined as keyword argument. Keyword argument value will be used")
                else:
                    kwargs[argvKey] = arg[argvKey]