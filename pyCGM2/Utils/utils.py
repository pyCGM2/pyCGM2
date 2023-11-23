import pyCGM2
import pyCGM2; LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional

class FrameConverter():
    def __init__(self,firstFrame:int,lastFrame:int, nppf:int):
        self.firstFrame = firstFrame
        self.lastFrame = lastFrame
        self.nppf = nppf

    def toAnalog(self,pointFrame:int,reframeToOne:bool=False):
        if not reframeToOne:
           val = (pointFrame-self.firstFrame)*self.nppf + self.firstFrame
        else: 
           val = (pointFrame-self.firstFrame)*self.nppf
        
        return int(val)

    def toPoint(self,analogFrame:int,reframeToOne:bool=False):
        if not reframeToOne:
            val =  (analogFrame-self.firstFrame)/self.nppf + self.firstFrame
        else:
            val = (analogFrame-self.firstFrame)/self.nppf 
        
        return int(val)


def toBool(text:str):
    """convert text to bool

    Args:
        text (str): text

    """
    return True if text == "True" else False

def isInRange(val:float, min:float, max:float):
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

def checkSimilarElement(listData:List):
    if(len(set(listData))==1):
        for it in set(listData):
            return it
        return True
    else:
        LOGGER.logger.error("[pyCGM2] items are different in the inputed list" )
        return False

def getSimilarElement(listData:List):
    out = []
    for it in set(listData):
        out = it
    return out


def homogeneizeArguments(argv:Dict,kwargs:Dict):
    for arg in argv:
        if isinstance(arg,dict):
            for argvKey in arg.keys():
                if argvKey in kwargs.keys():
                    LOGGER.logger.warning("The positional argument (%s) is already defined as keyword argument. Keyword argument value will be used")
                else:
                    kwargs[argvKey] = arg[argvKey]