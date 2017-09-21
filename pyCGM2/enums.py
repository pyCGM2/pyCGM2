# -*- coding: utf-8 -*-

from enum import Enum


def isEnumMember(member, enum):
    """
        check if member of an enum
    """

    flag = False
    for enumIt in enum:
           if enumIt == member:
                flag = True
    return flag




class motionMethod(Enum):
    """ Enum defining method uses for computing a segment pose """
    NoMotion = 0
    Determinist = 1
    Sodervisk = 2



class MomentProjection(Enum):
    """ Enum defining in which Segment expressed kinetics"""
    Global = 0
    Proximal = 1
    Distal = 2
    JCS = 3
    JCS_Dual =4


class HarringtonPredictor(Enum):
    """ Enum defining harrington's regression predictor"""
    Native = "full"
    PelvisWidth = "PWonly"
    LegLength = "LLonly"

class SegmentSide(Enum):
    """ Enum defining segment side"""
    Central = 0
    Left = 1
    Right = 2


class PlotType(Enum):
    """ Enum defining segment side"""
    DESCRIPTIVE = 0
    CONSISTENCY = 1
    MEAN_ONLY = 2
# --- enum used with Btk-Models
# obsolete
#class BspModel(Enum):
#    Dempster = "Dempster"
#    DempsterVicon = "DempsterVicon"
#    DeLeva = "DeLeva"
#
#class Sex(Enum):
#    Male = "M"
#    Female = "F"
#
#
#class InverseDynamicAlgo(Enum):
#    Quaternion = "quaternion"
#    Generic = "generic"
#    RotationMatrix = "rotationMatrix"
