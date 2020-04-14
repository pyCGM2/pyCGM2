# -*- coding: utf-8 -*-
from enum import Enum


def enumFromtext(memberStr, enum):
    if memberStr  in enum.__members__.keys():
        return enum.__members__[memberStr]
    else:
        raise Exception ("[pyCGM2] %s not found in targeted enum"%(memberStr))


def isValueBelongToEnum(value, enum):
    """

    """
    flag = False
    for it in enum.__members__.items():
        if it[1].value == value:
            flag = True
    return flag




def isEnumMember(member, enum):
    """
        check if member of an enum
    """

    flag = False
    for enumIt in enum:
           if enumIt == member:
                flag = True
    return flag


class DataType(Enum):
    """  """
    Marker = 0
    Angle = 1
    Segment = 3
    Moment = 4
    Force = 5
    Power = 6


class motionMethod(Enum):
    """ Enum defining method uses for computing a segment pose """
    Unknown = 0
    Determinist = 1
    Sodervisk = 2


class MomentProjection(Enum):
    """ Enum defining in which Segment expressed kinetics"""
    Global = "Global"
    Proximal = "Proximal"
    Distal = "Distal"
    JCS = "JCS"
    JCS_Dual ="JCS_Dual"


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


class EmgAmplitudeNormalization(Enum):
    """ Enum defining harrington's regression predictor"""
    MaxMax = "MaxMax"
    MeanMax = "MeanMax"
    MedianMax = "MedianMax"
    Threshold = "Threshold"

class BodyPart(Enum):
    LowerLimb=0
    LowerLimbTrunk=1
    FullBody=2
    UpperLimb=3


class JointCalibrationMethod(Enum):
    Basic = "lateralMarker"
    KAD = "KAD"
    Medial = "medial"

class BodyPartPlot(Enum):
    LowerLimb="LowerLimb"
    Trunk="Trunk"
    UpperLimb="UpperLimb"

class EclipseType(Enum):
    Session="Session.enf"
    Trial="Trial.enf"
    Patient="Patient.enf"

class AnalysisSection(Enum):
    Kinematic="Kinematic"
    Kinetic="Kinetic"
    Emg="Emg"
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
