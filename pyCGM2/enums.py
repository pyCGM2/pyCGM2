# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Enums
#APIDOC["Draft"]=False
#--end--
from enum import Enum


def enumFromtext(memberStr, enum):
    """get the enum value from text

    Args:
        memberStr (str): enum attribute label.
        enum (pyCGM2.Enum):  enum

    ```python
    enums.enumFromtext("Global",enums.MomentProjection)
    ```

    """
    if memberStr  in enum.__members__.keys():
        return enum.__members__[memberStr]
    else:
        raise Exception ("[pyCGM2] %s not found in targeted enum"%(memberStr))



class DataType(Enum):
    """Enum defining model ouput data type
    """
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
    """ Enum defining in which Segment is expressed kinetics"""
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
    """ Enum defining the body part of a model"""
    LowerLimb=0
    LowerLimbTrunk=1
    FullBody=2
    UpperLimb=3


class JointCalibrationMethod(Enum):
    """ Enum defining how a joint centre is calibrated"""

    Basic = "lateralMarker"
    KAD = "KAD"
    Medial = "medial"

class BodyPartPlot(Enum):
    """ Enum defining plot panel from a body part"""

    LowerLimb="LowerLimb"
    Trunk="Trunk"
    UpperLimb="UpperLimb"

class EclipseType(Enum):
    """Enum defining a Vicon Eclipse node
    """
    Session="Session.enf"
    Trial="Trial.enf"
    Patient="Patient.enf"

class AnalysisSection(Enum):
    """Enum defining a section of an `analysis` instance
    """
    Kinematic="Kinematic"
    Kinetic="Kinetic"
    Emg="Emg"
