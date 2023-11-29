
from enum import Enum


def enumFromtext(memberStr, enum):
    """
    Get the enum value from a text representation of an enum member.

    Args:
        memberStr (str): The text representation of the enum attribute label.
        enum (Enum): The enum class to search for the member.

    Returns:
        Enum Member: Corresponding enum member.

    Raises:
        Exception: If the memberStr is not found in the enum class.

    Example:
        ```python
        enums.enumFromtext("Global", enums.MomentProjection)
        ```
    """
    if memberStr  in enum.__members__.keys():
        return enum.__members__[memberStr]
    else:
        raise Exception ("[pyCGM2] %s not found in targeted enum"%(memberStr))



class DataType(Enum):
    """
    Enum defining model output data types.

    Attributes:
        Marker (int): Represents marker data type.
        Angle (int): Represents angle data type.
        Segment (int): Represents segment data type.
        Moment (int): Represents moment data type.
        Force (int): Represents force data type.
        Power (int): Represents power data type.
    """
    Marker = 0
    Angle = 1
    Segment = 3
    Moment = 4
    Force = 5
    Power = 6


class motionMethod(Enum):
    """
    Enum defining methods used for computing a segment pose.

    Attributes:
        Unknown (int): Unknown method.
        Determinist (int): Deterministic method.
        Sodervisk (int): Sodervisk method.
    """
    Unknown = 0
    Determinist = 1
    Sodervisk = 2


class MomentProjection(Enum):
    """
    Enum defining in which segment kinetics are expressed.

    Attributes:
        Global (str): Kinetics expressed in the global coordinate system.
        Proximal (str): Kinetics expressed in the proximal segment.
        Distal (str): Kinetics expressed in the distal segment.
        JCS (str): Kinetics expressed in the joint coordinate system.
        JCS_Dual (str): Kinetics expressed in the dual joint coordinate system.
    """
    Global = "Global"
    Proximal = "Proximal"
    Distal = "Distal"
    JCS = "JCS"
    JCS_Dual ="JCS_Dual"


class HarringtonPredictor(Enum):
    """
    Enum defining Harrington's regression predictors.

    Attributes:
        Native (str): Full regression predictor.
        PelvisWidth (str): Predictor based only on pelvis width.
        LegLength (str): Predictor based only on leg length.
    """
    Native = "full"
    PelvisWidth = "PWonly"
    LegLength = "LLonly"

class SegmentSide(Enum):
    """
    Enum defining segment sides.

    Attributes:
        Central (int): Central or midline segment.
        Left (int): Left side segment.
        Right (int): Right side segment.
    """
    Central = 0
    Left = 1
    Right = 2


class EmgAmplitudeNormalization(Enum):
    """
    Enum defining methods for EMG amplitude normalization.

    Attributes:
        MaxMax (str): Maximum of maximum normalization.
        MeanMax (str): Mean of maximum normalization.
        MedianMax (str): Median of maximum normalization.
        Threshold (str): Threshold-based normalization.
    """
    MaxMax = "MaxMax"
    MeanMax = "MeanMax"
    MedianMax = "MedianMax"
    Threshold = "Threshold"

class BodyPart(Enum):
    """
    Enum defining body parts of a model.

    Attributes:
        LowerLimb (int): Lower limb body part.
        LowerLimbTrunk (int): Lower limb and trunk body part.
        FullBody (int): Full body.
        UpperLimb (int): Upper limb body part.
    """
    LowerLimb=0
    LowerLimbTrunk=1
    FullBody=2
    UpperLimb=3


class JointCalibrationMethod(Enum):
    """
    Enum defining methods for joint center calibration.

    Attributes:
        Basic (str): Lateral marker-based calibration.
        KAD (str): Knee-ankle distance-based calibration.
        Medial (str): Medial marker-based calibration.
    """

    Basic = "lateralMarker"
    KAD = "KAD"
    Medial = "medial"

class BodyPartPlot(Enum):
    """
    Enum defining plot panels based on body parts.

    Attributes:
        LowerLimb (str): Lower limb plot panel.
        Trunk (str): Trunk plot panel.
        UpperLimb (str): Upper limb plot panel.
    """

    LowerLimb="LowerLimb"
    Trunk="Trunk"
    UpperLimb="UpperLimb"

class EclipseType(Enum):
    """
    Enum defining Vicon Eclipse node types.

    Attributes:
        Session (str): Session node type.
        Trial (str): Trial node type.
        Patient (str): Patient node type.
    """
    Session="Session.enf"
    Trial="Trial.enf"
    Patient="Patient.enf"

class AnalysisSection(Enum):
    """
    Enum defining sections of an `analysis` instance.

    Attributes:
        Kinematic (str): Kinematic section.
        Kinetic (str): Kinetic section.
        Emg (str): EMG section.
    """
    Kinematic="Kinematic"
    Kinetic="Kinetic"
    Emg="Emg"
