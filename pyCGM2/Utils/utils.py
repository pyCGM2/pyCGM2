import pyCGM2
import pyCGM2; LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional

class FrameConverter():
    """
    A class for converting frame numbers between point and analog data formats.

    Attributes:
        firstFrame (int): The first frame number in the dataset.
        lastFrame (int): The last frame number in the dataset.
        nppf (int): Number of analog points per frame.

    Methods:
        toAnalog(self, pointFrame, reframeToOne): Convert a point frame number to an analog frame number.
        toPoint(self, analogFrame, reframeToOne): Convert an analog frame number to a point frame number.
    """
    def __init__(self,firstFrame:int,lastFrame:int, nppf:int):
        """Initialize the FrameConverter with frame range and conversion factor"""
        self.firstFrame = firstFrame
        self.lastFrame = lastFrame
        self.nppf = nppf

    def toAnalog(self,pointFrame:int,reframeToOne:bool=False):
        """
        Convert a point frame number to an analog frame number.

        Args:
            pointFrame (int): The frame number in point format.
            reframeToOne (bool, optional): If True, reframe to start from one. Defaults to False.

        Returns:
            int: The corresponding frame number in analog format.
        """
        if not reframeToOne:
           val = (pointFrame-self.firstFrame)*self.nppf + self.firstFrame
        else: 
           val = (pointFrame-self.firstFrame)*self.nppf
        
        return int(val)

    def toPoint(self,analogFrame:int,reframeToOne:bool=False):
        """
        Convert an analog frame number to a point frame number.

        Args:
            analogFrame (int): The frame number in analog format.
            reframeToOne (bool, optional): If True, reframe to start from one. Defaults to False.

        Returns:
            int: The corresponding frame number in point format.
        """
        if not reframeToOne:
            val =  (analogFrame-self.firstFrame)/self.nppf + self.firstFrame
        else:
            val = (analogFrame-self.firstFrame)/self.nppf 
        
        return int(val)


def toBool(text:str):
    """
    Convert a text string to a boolean value.

    Args:
        text (str): The text to convert.

    Returns:
        bool: True if text is 'True', False otherwise.
    """
    return True if text == "True" else False

def isInRange(val:float, min:float, max:float):
    """
    Check if a value is within a specified range.

    Args:
        val (float): The value to check.
        min (float): The minimum value of the range.
        max (float): The maximum value of the range.

    Returns:
        bool: True if val is within the range, False otherwise.
    """

    if val<min or val>max:
        return False
    else:
        return True

def str(unicodeVariable):
    """
    Encode a unicode variable to a specified encoding format.

    Args:
        unicodeVariable: A variable containing unicode data.

    Returns:
        Encoded string in the specified encoding format of pyCGM2.
    """
    return unicodeVariable.encode(pyCGM2.ENCODER)

def checkSimilarElement(listData:List):
    """
    Check if all elements in a list are similar.

    Args:
        listData (List): The list to check for similarity.

    Returns:
        bool or element: Returns the similar element if all are the same, otherwise returns False.
    """
    if(len(set(listData))==1):
        for it in set(listData):
            return it
        return True
    else:
        LOGGER.logger.error("[pyCGM2] items are different in the inputed list" )
        return False

def getSimilarElement(listData:List):
    """
    Get a similar element from a list. If multiple different elements exist, returns one of them.

    Args:
        listData (List): The list to extract the similar element from.

    Returns:
        Element: A similar element from the list.
    """
    out = []
    for it in set(listData):
        out = it
    return out


def homogeneizeArguments(argv:Dict,kwargs:Dict):
    """
    Homogenize arguments by merging positional and keyword arguments into a single dictionary.

    Args:
        argv (Dict): Positional arguments as a dictionary.
        kwargs (Dict): Keyword arguments as a dictionary.
    """
    for arg in argv:
        if isinstance(arg,dict):
            for argvKey in arg.keys():
                if argvKey in kwargs.keys():
                    LOGGER.logger.warning("The positional argument (%s) is already defined as keyword argument. Keyword argument value will be used")
                else:
                    kwargs[argvKey] = arg[argvKey]