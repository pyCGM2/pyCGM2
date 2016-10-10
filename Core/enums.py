# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:24:38 2016

@author: aaa34169
"""

from enum import Enum


def isEnumMember(member, enum):

    flag = False
    for enumIt in enum:
           if enumIt == member:
                flag = True
    return flag




class motionMethod(Enum):
    NoMotion = 0
    Native = 1
    Sodervisk = 2


class MomentProjection(Enum):
    Global = 0
    Proximal = 1
    Distal = 2


class BspModel(Enum):
    Dempster = "Dempster"
    DempsterVicon = "DempsterVicon"    
    DeLeva = "DeLeva"

class Sex(Enum):
    Male = "M"
    Female = "F"
    

class InverseDynamicAlgo(Enum):
    Quaternion = "quaternion"
    Generic = "generic"    
    RotationMatrix = "rotationMatrix"


class HarringtonPredictor(Enum):
    Native = "full"
    PelvisWidth = "PWonly"    
    LegLength = "LLonly"

class SegmentSide(Enum):
    Central = 0
    Left = 1
    Right = 2