# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Model/CGM2
#APIDOC["Draft"]=False
#--end--
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
import copy

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")

from pyCGM2 import enums
from pyCGM2.Model import model, modelDecorator, frame, motion
from pyCGM2.Math import euler
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusTools


class CGM(model.Model):
    """
    Abstract Class of the Conventional Gait Model
    """
    KAD_MARKERS = {"Left" : ["LKAX","LKD1","LKD2"], "Right" : ["RKAX","RKD1","RKD2"]}

    ANALYSIS_KINEMATIC_LABELS_DICT ={
                            'Left': ["LHipAngles","LKneeAngles","LAnkleAngles","LFootProgressAngles","LPelvisAngles",
                                   "LForeFootAngles",
                                   "LThoraxAngles","LSpineAngles",
                                   "LNeckAngles","LHeadAngles",
                                   "LShoulderAngles","LElbowAngles","LWristAngles"],
                           'Right': ["RHipAngles","RKneeAngles","RAnkleAngles","RFootProgressAngles","RPelvisAngles",
                                    "RForeFootAngles",
                                    "RThoraxAngles","RSpineAngles",
                                    "RNeckAngles","RHeadAngles",
                                    "RShoulderAngles","RElbowAngles","RWristAngles"]}

    ANALYSIS_KINETIC_LABELS_DICT ={
                            'Left': ["LHipMoment","LKneeMoment","LAnkleMoment","LHipPower","LKneePower","LAnklePower"],
                            'Right': ["RHipMoment","RKneeMoment","RAnkleMoment","RHipPower","RKneePower","RAnklePower"]}

    VERSIONS = ["CGM1", "CGM1.1", "CGM2.1",  "CGM2.2", "CGM2.3", "CGM2.4", "CGM2.5"]


    def __init__(self):

        super(CGM, self).__init__()
        self.m_useLeftTibialTorsion=False
        self.m_useRightTibialTorsion=False
        self.staExpert= False

        self.m_staticTrackingMarkers = None

    def setSTAexpertMode(self,boolFlag):
        self.staExpert= boolFlag

    def setStaticTrackingMarkers(self,markers):
        """Set tracking markers

        Args:
            markers (list): tracking markers

        """


        self.m_staticTrackingMarkers = markers

    def getStaticTrackingMarkers(self):
        """get tracking markers"""

        return self.m_staticTrackingMarkers

    @classmethod
    def detectCalibrationMethods(cls, acqStatic):
        """ *Class method* to detect the method used to calibrate knee and ankle joint centres

        Args:
            acqStatic (btk.Acquisition): acquisition.

        """
        # Left knee
        LKnee = enums.JointCalibrationMethod.Basic
        if btkTools.isPointsExist(acqStatic,["LKAX","LKD1","LKD2"]):
            LKnee = enums.JointCalibrationMethod.KAD
        elif btkTools.isPointsExist(acqStatic,["LKNM","LKNE"]):
            LKnee = enums.JointCalibrationMethod.Medial

        # right knee
        RKnee = enums.JointCalibrationMethod.Basic
        if btkTools.isPointsExist(acqStatic,["RKAX","RKD1","RKD2"]):
            RKnee = enums.JointCalibrationMethod.KAD
        elif btkTools.isPointsExist(acqStatic,["RKNM","RKNE"]):
            RKnee = enums.JointCalibrationMethod.Medial

        # Left ankle
        LAnkle = enums.JointCalibrationMethod.Basic
        if btkTools.isPointsExist(acqStatic,["LANK","LMED"]):
            LAnkle = enums.JointCalibrationMethod.Medial

        # right ankle
        RAnkle = enums.JointCalibrationMethod.Basic
        if btkTools.isPointsExist(acqStatic,["RANK","RMED"]):
            RAnkle = enums.JointCalibrationMethod.Medial

        dectectedCalibrationMethods=dict()
        dectectedCalibrationMethods["Left Knee"] = LKnee
        dectectedCalibrationMethods["Right Knee"] = RKnee
        dectectedCalibrationMethods["Left Ankle"] = LAnkle
        dectectedCalibrationMethods["Right Ankle"] = RAnkle

        return dectectedCalibrationMethods

    @classmethod
    def get_markerLabelForPiGStatic(cls,dcm):
        """ *Class method* returning marker label of the knee and ankle joint centres

        Args:
            dcm (dict): dictionary returned from the function `detectCalibrationMethods`

        """

        useLeftKJCmarkerLabel = "LKJC"
        useLeftAJCmarkerLabel = "LAJC"
        useRightKJCmarkerLabel = "RKJC"
        useRightAJCmarkerLabel = "RAJC"


        # KAD - kadMed
        if dcm["Left Knee"] == enums.JointCalibrationMethod.KAD:
            useLeftKJCmarkerLabel = "LKJC_KAD"
            useLeftAJCmarkerLabel = "LAJC_KAD"
            if  dcm["Left Ankle"] == enums.JointCalibrationMethod.Medial:
                useLeftAJCmarkerLabel = "LAJC_MID"

        if dcm["Right Knee"] == enums.JointCalibrationMethod.KAD:
            useRightKJCmarkerLabel = "RKJC_KAD"
            useRightAJCmarkerLabel = "RAJC_KAD"
            if  dcm["Right Ankle"] == enums.JointCalibrationMethod.Medial:
                useRightAJCmarkerLabel = "RAJC_MID"


        return [useLeftKJCmarkerLabel,useLeftAJCmarkerLabel,useRightKJCmarkerLabel,useRightAJCmarkerLabel]


class CGM1(CGM):
    """
    Conventional Gait Model 1 (aka Vicon Plugin Gait Clone)
    """

    LOWERLIMB_TRACKING_MARKERS=["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]
    THORAX_TRACKING_MARKERS=["C7", "T10","CLAV", "STRN"]
    UPPERLIMB_TRACKING_MARKERS=[ "LSHO",  "LELB", "LWRA", "LWRB",  "LFIN","RSHO", "RELB", "RWRA", "RWRB",  "RFIN", "LFHD","LBHD","RFHD","RBHD"]

    LOWERLIMB_SEGMENTS=["Pelvis", "Left Thigh","Left Shank", "Left Shank Proximal","Left Foot","Right Thigh","Right Shank","Right Shank Proximal","Right Foot"]
    THORAX_SEGMENTS=["Thorax"]
    UPPERLIMB_SEGMENTS=["Head", "Thorax","Left Clavicle", "Left UpperArm","Left ForeArm","Left Hand","Right Clavicle", "Right UpperArm","Right ForeArm","Right Hand"]


    LOWERLIMB_JOINTS=["LHip", "LKnee","LAnkle", "RHip", "RKnee","RAnkle"]
    THORAX_JOINTS=["LSpine","RSpine"]
    UPPERLIMB_JOINTS=["LShoulder", "LElbow","LWrist", "LNeck","RShoulder", "RElbow","RWrist", "RNeck"]


    def __init__(self):

        super(CGM1, self).__init__()
        self.decoratedModel = False
        self.version = "CGM1.0"


        # init of few mp_computed
        self.mp_computed["LeftKneeFuncCalibrationOffset"] = 0
        self.mp_computed["RightKneeFuncCalibrationOffset"] = 0

        self._R_leftUnCorrfoot_dist_prox = np.eye(3,3)
        self._R_rightUnCorrfoot_dist_prox = np.eye(3,3)

    def setVersion(self,string):
        """ amend model vesion"""
        self.version = string

    def __repr__(self):
        return "CGM1.0"

    def _lowerLimbTrackingMarkers(self):
        return CGM1.LOWERLIMB_TRACKING_MARKERS#["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

    def _trunkTrackingMarkers(self):
        return CGM1.THORAX_TRACKING_MARKERS#["C7", "T10","CLAV", "STRN"]

    def _upperLimbTrackingMarkers(self):
        return CGM1.THORAX_TRACKING_MARKERS+CGM1.UPPERLIMB_TRACKING_MARKERS#S#["C7", "T10","CLAV", "STRN", "LELB", "LWRA", "LWRB", "LFRM", "LFIN", "RELB", "RWRA", "RWRB", "RFRM", "RFIN"]


    def getTrackingMarkers(self,acq):
        """return tracking markers

        Args:
            acq (btk.Acquisition): acquisition.

        """

        tracking_markers =  self._lowerLimbTrackingMarkers() + self._upperLimbTrackingMarkers()

        return tracking_markers

    def getStaticMarkers(self,dcm):
        """return static markers

        Args:
            dcm (dict): dictionary returned from the function `detectCalibrationMethods`

        """
        static_markers = self.getTrackingMarkers()

        if dcm["Left Knee"] == enums.JointCalibrationMethod.KAD:
            static_markers.remove("LKNE")
            static_markers = static_markers + CGM.KAD_MARKERS["Left"]
        elif dcm["Left Knee"] == enums.JointCalibrationMethod.Medial:
            static_markers.append("LMED")

        if dcm["Right Knee"] == enums.JointCalibrationMethod.KAD:
            static_markers.remove("RKNE")
            static_markers = static_markers + CGM.KAD_MARKERS["Right"]
        elif dcm["Right Knee"] == enums.JointCalibrationMethod.Medial:
            static_markers.append("RKNM")

        if dcm["Left Ankle"] == enums.JointCalibrationMethod.Medial:
            static_markers.append("LMED")
        if dcm["Right Ankle"] == enums.JointCalibrationMethod.Medial:
            static_markers.append("RMED")

        return static_markers


    def configure(self,detectedCalibrationMethods=None):
        """" configure the model

        Args:
            detectedCalibrationMethods (dict,optional[None]): dictionary returned from the function `detectCalibrationMethods`

        """

        bodyPart = enums.BodyPart.FullBody
        self.setBodyPart(bodyPart)
        LOGGER.logger.info("BodyPart found : %s" %(bodyPart.name))

        if detectedCalibrationMethods is not None:
            if detectedCalibrationMethods["Left Knee"] == enums.JointCalibrationMethod.KAD:
                if "LKNE" in self._lowerLimbTrackingMarkers(): self._lowerLimbTrackingMarkers().remove("LKNE")
            if detectedCalibrationMethods["Right Knee"] == enums.JointCalibrationMethod.KAD:
                if "RKNE" in self._lowerLimbTrackingMarkers(): self._lowerLimbTrackingMarkers().remove("RKNE")

        self._lowerLimbTrackingMarkers()+["LKNE","RKNE"]

        self._lowerlimbConfigure()
        self._trunkConfigure()
        self._upperLimbConfigure()

        self._coordinateSystemDefinitions()

    def _lowerlimbConfigure(self):
        self.addSegment("Pelvis",0,enums.SegmentSide.Central,calibration_markers=[], tracking_markers = ["LASI","RASI","LPSI","RPSI"])
        self.addSegment("Left Thigh",1,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LKNE","LTHI"])
        self.addSegment("Right Thigh",4,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RKNE","RTHI"])
        self.addSegment("Left Shank",2,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LANK","LTIB"])
        self.addSegment("Left Shank Proximal",7,enums.SegmentSide.Left,cloneOf=True) # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Right Shank",5,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RANK","RTIB"])
        self.addSegment("Right Shank Proximal",8,enums.SegmentSide.Right,cloneOf=True)        # copy of Left Shank with anatomical frame modified by a tibial Rotation Value ( see calibration)
        self.addSegment("Left Foot",3,enums.SegmentSide.Left,calibration_markers=[""], tracking_markers = ["LHEE","LTOE"] )
        self.addSegment("Right Foot",6,enums.SegmentSide.Right,calibration_markers=[""], tracking_markers = ["RHEE","RTOE"])

        self.addChain("Left Lower Limb", [3,2,1,0]) # Dist ->Prox Todo Improve
        self.addChain("Right Lower Limb", [6,5,4,0])

        self.addJoint("LHip","Pelvis", "Left Thigh","YXZ","LHJC")
        if self.version == "CGM1.0":
            self.addJoint("LKnee","Left Thigh", "Left Shank Proximal","YXZ","LKJC")
        else:
            self.addJoint("LKnee","Left Thigh", "Left Shank","YXZ","LKJC")


        #self.addJoint("LKneeAngles_cgm","Left Thigh", "Left Shank","YXZ")
        self.addJoint("LAnkle","Left Shank", "Left Foot","YXZ","LAJC")

        self.addJoint("RHip","Pelvis", "Right Thigh","YXZ","RHJC")
        if self.version == "CGM1.0":
            self.addJoint("RKnee","Right Thigh", "Right Shank Proximal","YXZ","RKJC")
        else:
            self.addJoint("RKnee","Right Thigh", "Right Shank","YXZ","RKJC")


        #self.addJoint("RKneeAngles_cgm","Right Thigh", "Right Shank","YXZ")
        self.addJoint("RAnkle","Right Shank", "Right Foot","YXZ","RAJC")

        # clinics
        self.setClinicalDescriptor("LHip",enums.DataType.Angle, [0,1,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("LKnee",enums.DataType.Angle, [0,1,2],[+1.0,-1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("LAnkle",enums.DataType.Angle, [0,2,1],[-1.0,-1.0,-1.0], [ np.radians(90),0.0,0.0])
        self.setClinicalDescriptor("RHip",enums.DataType.Angle, [0,1,2],[-1.0,+1.0,+1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RKnee",enums.DataType.Angle, [0,1,2],[+1.0,+1.0,+1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RAnkle",enums.DataType.Angle, [0,2,1],[-1.0,+1.0,+1.0], [ np.radians(90),0.0,0.0])

        self.setClinicalDescriptor("LPelvis",enums.DataType.Angle,[0,1,2],[1.0,1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RPelvis",enums.DataType.Angle,[0,1,2],[1.0,-1.0,+1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("Left Foot",enums.DataType.Angle,[0,2,1],[1.0,1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("Right Foot",enums.DataType.Angle,[0,2,1],[1.0,-1.0,1.0], [0.0,0.0,0.0])

        # distal Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)

        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[-1.0,+1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,2,0],[1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[-1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Distal)

        # proximal Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)

        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[+1.0,+1.0,+1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Proximal)

        # Global Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)

        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[+1.0,+1.0,+1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,0,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,0,2],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.Global)

        # JCS Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)


        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,2,0],[1.0,-1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS)

        # JCS-dual Projection
        self.setClinicalDescriptor("LHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("LHip",enums.DataType.Moment, [1,0,2],[1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RHip",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RHip",enums.DataType.Moment, [1,0,2],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)

        self.setClinicalDescriptor("LKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("LKnee",enums.DataType.Moment, [1,0,2],[-1.0,-1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RKnee",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RKnee",enums.DataType.Moment, [1,0,2],[-1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)


        self.setClinicalDescriptor("LAnkle",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("LAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,-1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Force, [0,1,2],[-1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)
        self.setClinicalDescriptor("RAnkle",enums.DataType.Moment, [1,2,0],[1.0,1.0,1.0], [0.0,0.0,0.0],projection = enums.MomentProjection.JCS_Dual)

    def _trunkConfigure(self):
        self.addSegment("Thorax",0,enums.SegmentSide.Central,calibration_markers=[], tracking_markers = ["CLAV","C7","T10","STRN"])

        self.addJoint("LSpine","Thorax","Pelvis", "YXZ","LSJC")
        self.addJoint("RSpine","Thorax","Pelvis", "YXZ","LSJC")

        self.setClinicalDescriptor("LSpine",enums.DataType.Angle, [0,1,2],[1.0,-1.0,-1.0], [np.radians(-180),0.0,np.radians(180)])
        self.setClinicalDescriptor("RSpine",enums.DataType.Angle, [0,1,2],[1.0,1.0,1.0], [np.radians(-180),0.0,np.radians(180)])
        self.setClinicalDescriptor("LThorax",enums.DataType.Angle,[0,1,2],[1.0,-1.0,1.0], [-np.radians(180),0.0,-np.radians(180)])
        self.setClinicalDescriptor("RThorax",enums.DataType.Angle,[0,1,2],[1.0,1.0,-1.0], [-np.radians(180),0.0,-np.radians(180)])

    def _upperLimbConfigure(self):
        self.addSegment("Head",0,enums.SegmentSide.Central,calibration_markers=["C7"], tracking_markers = ["LFHD","RFHD","LBHD","RBHD"])

        self.addSegment("Left Clavicle",0,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = [])
        self.addSegment("Left UpperArm",0,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LSJC","LELB"])
        self.addSegment("Left ForeArm",0,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LWRA","LWRB","LEJC"])
        self.addSegment("Left Hand",0,enums.SegmentSide.Left,calibration_markers=[], tracking_markers = ["LWRA","LWRB","LFIN","LWJC"])

        self.addSegment("Right Clavicle",0,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = [])
        self.addSegment("Right UpperArm",0,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RSJC","RELB"])
        self.addSegment("Right ForeArm",0,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RWRA","RWRB","REJC"])
        self.addSegment("Right Hand",0,enums.SegmentSide.Right,calibration_markers=[], tracking_markers = ["RWRA","RWRB","RFIN","RWJC"])



        self.addJoint("LShoulder","Thorax", "Left UpperArm","XYZ","LSJC")
        self.addJoint("LElbow","Left UpperArm", "Left ForeArm","YXZ","LEJC")
        self.addJoint("LWrist","Left ForeArm", "Left Hand","YXZ","LWJC")
        self.addJoint("LNeck","Head", "Thorax","YXZ","OT")

        self.addJoint("RShoulder","Thorax", "Right UpperArm","XYZ","RSJC")
        self.addJoint("RElbow","Right UpperArm", "Right ForeArm","YXZ","REJC")
        self.addJoint("RWrist","Right ForeArm", "Right Hand","YXZ","RWJC")
        self.addJoint("RNeck","Head", "Thorax","YXZ","OT")


        # clinics
        self.setClinicalDescriptor("LShoulder",enums.DataType.Angle, [1,0,2],[-1.0,1.0,-1.0], [0.0,np.radians(180),np.radians(-180)])
        self.setClinicalDescriptor("LElbow",enums.DataType.Angle, [0,2,1],[1.0,1.0,1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("LWrist",enums.DataType.Angle, [0,1,2],[1.0,-1.0,-1.0], [0.0,0,0.0])
        self.setClinicalDescriptor("LNeck",enums.DataType.Angle, [0,1,2],[-1.0,1.0,1.0], [0,np.radians(180),0.0])

        self.setClinicalDescriptor("RShoulder",enums.DataType.Angle, [1,0,2],[-1.0,-1.0,1.0], [0.0,-np.radians(180),np.radians(180)]) # warning. i got offset on the int/ext rotation i fixed with a special behaviour of ClinicalDescriptor
        self.setClinicalDescriptor("RElbow",enums.DataType.Angle, [0,2,1],[1.0,1.0,1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RWrist",enums.DataType.Angle, [0,1,2],[1.0,1.0,1.0], [0.0,0,0.0])
        self.setClinicalDescriptor("RNeck",enums.DataType.Angle, [0,1,2],[-1.0,1.0,-1.0], [-np.radians(180),0,np.radians(180)])

        #self.setClinicalDescriptor("LThorax",enums.DataType.Angle,[0,1,2],[1.0,1.0,1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("LHead",enums.DataType.Angle,[0,1,2],[-1.0,1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RHead",enums.DataType.Angle,[0,1,2],[-1.0,-1.0,1.0], [0.0,0.0,0.0])

    def calibrationProcedure(self):
        """
        Define the calibration Procedure

        Return:
            dict:  markers and sequence use for building Technical coordinate system
            dict:  markers and sequence use for building Anatomical coordinate system
        """

        dictRef={}
        dictRefAnatomical={}

        self._lowerLimbCalibrationProcedure(dictRef,dictRefAnatomical)
        self._trunkCalibrationProcedure(dictRef,dictRefAnatomical)
        self._upperLimbCalibrationProcedure(dictRef,dictRefAnatomical)

        return dictRef,dictRefAnatomical

    def _lowerLimbCalibrationProcedure(self,dictRef,dictRefAnatomical):
        dictRef["Pelvis"]={"TF" : {'sequence':"YZX", 'labels':   ["RASI","LASI","SACR","midASIS"]} }
        dictRef["Left Thigh"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LKNE","LHJC","LTHI","LKNE"]} }
        dictRef["Right Thigh"]={"TF" : {'sequence':"ZXY", 'labels':   ["RKNE","RHJC","RTHI","RKNE"]} }
        dictRef["Left Shank"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LANK","LKJC","LTIB","LANK"]} }
        dictRef["Right Shank"]={"TF" : {'sequence':"ZXY", 'labels':   ["RANK","RKJC","RTIB","RANK"]} }

        dictRef["Left Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LTOE","LAJC",None,"LAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis
        dictRef["Right Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["RTOE","RAJC",None,"RAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis

        dictRefAnatomical["Pelvis"]= {'sequence':"YZX", 'labels':  ["RASI","LASI","SACR","midASIS"]} # normaly : midHJC
        dictRefAnatomical["Left Thigh"]= {'sequence':"ZXiY", 'labels':  ["LKJC","LHJC","LKNE","LHJC"]} # origin = Proximal ( differ from native)
        dictRefAnatomical["Right Thigh"]= {'sequence':"ZXY", 'labels': ["RKJC","RHJC","RKNE","RHJC"]}
        dictRefAnatomical["Left Shank"]={'sequence':"ZXiY", 'labels':   ["LAJC","LKJC","LANK","LKJC"]}
        dictRefAnatomical["Right Shank"]={'sequence':"ZXY", 'labels':  ["RAJC","RKJC","RANK","RKJC"]}

        dictRefAnatomical["Left Foot"]={'sequence':"ZXiY", 'labels':  ["LTOE","LHEE",None,"LAJC"]}    # corrected foot
        dictRefAnatomical["Right Foot"]={'sequence':"ZXiY", 'labels':  ["RTOE","RHEE",None,"RAJC"]}    # corrected foot

    def _trunkCalibrationProcedure(self,dictRef,dictRefAnatomical):
        dictRef["Thorax"]={"TF" : {'sequence':"ZYX", 'labels':   ["midTop","midBottom","midFront","CLAV"]} }
        dictRefAnatomical["Thorax"]= {'sequence':"ZYX", 'labels':  ["midTop","midBottom","midFront","OT"]}

    def _upperLimbCalibrationProcedure(self,dictRef,dictRefAnatomical):

        dictRef["Left Clavicle"]={"TF" : {'sequence':"ZXY", 'labels':   ["LSJC","OT","LVWM","LSJC"]} } # OT and LWM from thorax
        dictRef["Right Clavicle"]={"TF" : {'sequence':"ZXY", 'labels':   ["RSJC","OT","RVWM","RSJC"]} } # OT and LWM from thorax
        dictRef["Head"]={"TF" : {'sequence':"XZY", 'labels':   ["HC","midFH","midLH","midFH"]} }
        dictRef["Left UpperArm"]={"TF" : {'sequence':"ZYiX", 'labels':   ["LELB","LSJC","LCVM","LELB"]} }
        dictRef["Left ForeArm"]={"TF" : {'sequence':"ZXY", 'labels':   ["LWRA","LEJC","LWRB","LWRB"]} }
        dictRef["Left Hand"]={"TF" : {'sequence':"ZYX", 'labels':   ["LFIN","LWJC","LMWP","LFIN"]} }
        dictRef["Right UpperArm"]={"TF" : {'sequence':"ZYiX", 'labels':   ["RELB","RSJC","RCVM","RELB"]} }
        dictRef["Right ForeArm"]={"TF" : {'sequence':"ZXY", 'labels':   ["RWRA","REJC","RWRB","RWRB"]} }
        dictRef["Right Hand"]={"TF" : {'sequence':"ZYX", 'labels':   ["RFIN","RWJC","RMWP","RFIN"]} }


        dictRefAnatomical["Left Clavicle"]={'sequence':"ZXY", 'labels':   ["LSJC","OT","LVWM","LSJC"]} # idem technical
        dictRefAnatomical["Right Clavicle"]={'sequence':"ZXY", 'labels':   ["RSJC","OT","RVWM","RSJC"]} # idem technical
        dictRefAnatomical["Head"]={'sequence':"XZY", 'labels':   ["HC","midFH","midLH","midFH"]}
        dictRefAnatomical["Left UpperArm"]={'sequence':"ZYiX", 'labels':   ["LEJC","LSJC","LWJC","LSJC"]}
        dictRefAnatomical["Left ForeArm"]={'sequence':"ZXiY", 'labels':   ["LWJC","LEJC",None,"LEJC"]} # used y axis of upper
        dictRefAnatomical["Left Hand"]={'sequence':"ZYX", 'labels':   ["LHO","LWJC","LMWP","LWJC"]}
        dictRefAnatomical["Right UpperArm"]={'sequence':"ZYiX", 'labels':   ["REJC","RSJC","RWJC","RSJC"]}
        dictRefAnatomical["Right ForeArm"]={'sequence':"ZXiY", 'labels':   ["RWJC","REJC",None,"REJC"]} # used y axis of upper
        dictRefAnatomical["Right Hand"]={'sequence':"ZYX", 'labels':   ["RHO","RWJC","RMWP","LWJC"]}

    def _lowerLimbCoordinateSystemDefinitions(self):
        self.setCoordinateSystemDefinition( "Pelvis", "PELVIS", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Thigh", "LFEMUR", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Thigh", "RFEMUR", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Shank", "LTIBIA", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Shank", "RTIBIA", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Shank Proximal", "LTIBIAPROX", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Shank Proximal", "RTIBIAPROX", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Foot", "LFOOT", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Foot", "RFOOT", "Anatomic")

    def _trunkCoordinateSystemDefinitions(self):
        self.setCoordinateSystemDefinition( "Thorax", "THORAX", "Anatomic")

    def _upperLimbCoordinateSystemDefinitions(self):
        self.setCoordinateSystemDefinition( "Left Clavicle", "LCLAVICLE", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Clavicle", "RCLAVICLE", "Anatomic")
        self.setCoordinateSystemDefinition( "Head", "HEAD", "Anatomic")
        self.setCoordinateSystemDefinition( "Left UpperArm", "LUPPERARM", "Anatomic")
        self.setCoordinateSystemDefinition( "Left ForeArm", "LFOREARM", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Hand", "LHANDARM", "Anatomic")
        self.setCoordinateSystemDefinition( "Right UpperArm", "RUPPERARM", "Anatomic")
        self.setCoordinateSystemDefinition( "Right ForeArm", "RFOREARM", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Hand", "RHANDARM", "Anatomic")


    def _coordinateSystemDefinitions(self):

        self._lowerLimbCoordinateSystemDefinitions()
        self._trunkCoordinateSystemDefinitions()
        self._upperLimbCoordinateSystemDefinitions()

    def calibrate(self, aquiStatic, dictRef, dictAnatomic, options=None):
        """calibrate the model

        Args:
            aquiStatic (btk.acquisition): acquisition
            dictRef (dict): markers and sequence used for building the technical coordinate system
            dictAnatomic (dict): markers and sequence used for building the anatomical coordinate system
            options (dict, optional[None]): passed arguments to embedded methods

        """
        #TODO : to input Frane init and Frame end manually

        LOGGER.logger.debug("=====================================================")
        LOGGER.logger.debug("===================CGM CALIBRATION===================")
        LOGGER.logger.debug("=====================================================")

        ff=aquiStatic.GetFirstFrame()
        lf=aquiStatic.GetLastFrame()


        frameInit=ff-ff
        frameEnd=lf-ff+1

        if not self.decoratedModel:
            LOGGER.logger.debug(" Native CGM")
            if not btkTools.isPointExist(aquiStatic,"LKNE"):
                btkTools.smartAppendPoint(aquiStatic,"LKNE",np.zeros((aquiStatic.GetPointFrameNumber(),3) ))
            if not btkTools.isPointExist(aquiStatic,"RKNE"):
                btkTools.smartAppendPoint(aquiStatic,"RKNE",np.zeros((aquiStatic.GetPointFrameNumber(),3) ))

        else:
            LOGGER.logger.debug(" Decorated CGM")

        # ---- Pelvis-THIGH-SHANK CALIBRATION
        #-------------------------------------
        self._pelvis_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)

        self._left_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
        self._right_thigh_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)

        self._left_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)
        self._right_shank_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)

        # calibration of anatomical Referentials
        self._pelvis_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

        self._left_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self._right_thigh_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)


        if "LeftThighRotation" in self.mp and self.mp["LeftThighRotation"] != 0:
            self.mp_computed["LeftThighRotationOffset"]= self.mp["LeftThighRotation"]
        else:
            self.getThighOffset(side="left")

        # management of Functional method
        if self.mp_computed["LeftKneeFuncCalibrationOffset"] != 0:
            offset = self.mp_computed["LeftKneeFuncCalibrationOffset"]
            # SARA
            if self.checkCalibrationProperty("LeftFuncKneeMethod","SARA"):
                LOGGER.logger.debug("Left knee functional calibration : SARA ")
            # 2DOF
            elif self.checkCalibrationProperty("LeftFuncKneeMethod","2DOF"):
                LOGGER.logger.debug("Left knee functional calibration : 2Dof ")
            self._rotateAnatomicalFrame("Left Thigh",offset,
                                        aquiStatic, dictAnatomic,frameInit,frameEnd)


        if "RightThighRotation" in self.mp and self.mp["RightThighRotation"] != 0:
            self.mp_computed["RightThighRotationOffset"]= self.mp["RightThighRotation"]
        else:
            self.getThighOffset(side="right")

        # management of Functional method
        if self.mp_computed["RightKneeFuncCalibrationOffset"] != 0:
            offset = self.mp_computed["RightKneeFuncCalibrationOffset"]
            # SARA
            if self.checkCalibrationProperty("RightFuncKneeMethod","SARA"):
                LOGGER.logger.debug("Left knee functional calibration : SARA ")
            # 2DOF
            elif self.checkCalibrationProperty("RightFuncKneeMethod","2DOF"):
                LOGGER.logger.debug("Left knee functional calibration : 2Dof ")

            self._rotateAnatomicalFrame("Right Thigh",offset,
                                        aquiStatic, dictAnatomic,frameInit,frameEnd)




        self._left_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)
        self._right_shank_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

        # shakRotation
        if "LeftShankRotation" in self.mp and self.mp["LeftShankRotation"] != 0:
            self.mp_computed["LeftShankRotationOffset"]= self.mp["LeftShankRotation"]
        else:
            self.getShankOffsets(side="left")

        if "RightShankRotation" in self.mp and self.mp["RightShankRotation"] != 0:
            self.mp_computed["RightShankRotationOffset"]= self.mp["RightShankRotation"]
        else:
            self.getShankOffsets(side="right")

        # tibial Torsion
        if "LeftTibialTorsion" in self.mp and self.mp["LeftTibialTorsion"] != 0:
            self.mp_computed["LeftTibialTorsionOffset"]= self.mp["LeftTibialTorsion"]
            self.m_useLeftTibialTorsion=True
        else:
            if self.m_useLeftTibialTorsion:
                self.getTibialTorsionOffset(side="left")
            else:
                self.mp_computed["LeftTibialTorsionOffset"]= 0

        #   right
        if "RightTibialTorsion" in self.mp and self.mp["RightTibialTorsion"] != 0:
            self.mp_computed["RightTibialTorsionOffset"]= self.mp["RightTibialTorsion"]
            self.m_useRightTibialTorsion=True
        else:
            if self.m_useRightTibialTorsion:
                self.getTibialTorsionOffset(side="right")
            else:
                self.mp_computed["RightTibialTorsionOffset"]= 0


        # AbdAdd offset
        self.getAbdAddAnkleJointOffset(side="left")
        self.getAbdAddAnkleJointOffset(side="right")

        #   shank Prox ( copy )
        self.updateSegmentFromCopy("Left Shank Proximal", self.getSegment("Left Shank")) # look out . I copied the shank instance and rename it
        self._left_shankProximal_AnatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options) # alter static Frame


        self.updateSegmentFromCopy("Right Shank Proximal", self.getSegment("Right Shank"))
        self._right_shankProximal_AnatomicalCalibrate(aquiStatic,dictAnatomic,frameInit,frameEnd,options=options) # alter static Frame

        # ---- FOOT CALIBRATION
        #-------------------------------------
        # foot ( need  Y-axis of the shank anatomic Frame)
        self._left_unCorrectedFoot_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)

        self._left_foot_corrected_calibrate(aquiStatic, dictAnatomic,frameInit,frameEnd,options=options)

        self._right_unCorrectedFoot_calibrate(aquiStatic, dictRef,frameInit,frameEnd,options=options)

        self._right_foot_corrected_calibrate(aquiStatic, dictAnatomic,frameInit,frameEnd,options=options)

        self.getFootOffset(side = "both")


        self._torso_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
        self._torso_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

        self._head_calibrate(aquiStatic,dictRef,frameInit,frameEnd,options=options)
        self._head_AnatomicalCalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

        self._clavicle_calibrate("Left",aquiStatic,dictRef,frameInit,frameEnd,options=options)
        self._clavicle_Anatomicalcalibrate("Left",aquiStatic, dictAnatomic,frameInit,frameEnd)

        self._constructArmVirtualMarkers("Left", aquiStatic)

        self._upperArm_calibrate("Left",aquiStatic, dictRef,frameInit,frameEnd, options=options)
        self._foreArm_calibrate("Left",aquiStatic, dictRef,frameInit,frameEnd, options=options)

        self._upperArm_Anatomicalcalibrate("Left",aquiStatic, dictAnatomic,frameInit,frameEnd)
        self._foreArm_Anatomicalcalibrate("Left",aquiStatic, dictAnatomic,frameInit,frameEnd)

        self._hand_calibrate("Left",aquiStatic, dictRef,frameInit,frameEnd, options=options)
        self._hand_Anatomicalcalibrate("Left",aquiStatic, dictAnatomic,frameInit,frameEnd)

        self._clavicle_calibrate("Right",aquiStatic,dictRef,frameInit,frameEnd,options=options)
        self._clavicle_Anatomicalcalibrate("Right",aquiStatic, dictAnatomic,frameInit,frameEnd)

        self._constructArmVirtualMarkers("Right", aquiStatic)

        self._upperArm_calibrate("Right",aquiStatic, dictRef,frameInit,frameEnd, options=options)
        self._foreArm_calibrate("Right",aquiStatic, dictRef,frameInit,frameEnd, options=options)

        self._upperArm_Anatomicalcalibrate("Right",aquiStatic, dictAnatomic,frameInit,frameEnd)
        self._foreArm_Anatomicalcalibrate("Right",aquiStatic, dictAnatomic,frameInit,frameEnd)

        self._hand_calibrate("Right",aquiStatic, dictRef,frameInit,frameEnd, options=options)
        self._hand_Anatomicalcalibrate("Right",aquiStatic, dictAnatomic,frameInit,frameEnd)


    # ---- Technical Referential Calibration
    def _pelvis_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):

        pfn = aquiStatic.GetPointFrameNumber()

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment("Pelvis")
        seg.resetMarkerLabels()

        # ---  additional markers and Update of the marker segment list

        # new markers
        valSACR=(aquiStatic.GetPoint("LPSI").GetValues() + aquiStatic.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"SACR",valSACR,desc="")

        valMidAsis=(aquiStatic.GetPoint("LASI").GetValues() + aquiStatic.GetPoint("RASI").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"midASIS",valMidAsis,desc="")

        seg.addCalibrationMarkerLabel("SACR")
        seg.addCalibrationMarkerLabel("midASIS")


        # new mp
        if "PelvisDepth" in self.mp and self.mp["PelvisDepth"] != 0:
            LOGGER.logger.debug("PelvisDepth defined from your vsk file")
            self.mp_computed["PelvisDepth"] = self.mp["PelvisDepth"]
        else:
            LOGGER.logger.debug("Pelvis Depth computed and added to model parameters")
            self.mp_computed["PelvisDepth"] = np.linalg.norm( valMidAsis[frameInit:frameEnd,:].mean(axis=0)-valSACR[frameInit:frameEnd,:].mean(axis=0)) - 2.0* (markerDiameter/2.0) -2.0* (basePlate/2.0)

        if "InterAsisDistance" in self.mp and self.mp["InterAsisDistance"] != 0:
            LOGGER.logger.debug("InterAsisDistance defined from your vsk file")
            self.mp_computed["InterAsisDistance"] = self.mp["InterAsisDistance"]
        else:
            LOGGER.logger.debug("asisDistance computed and added to model parameters")
            self.mp_computed["InterAsisDistance"] = np.linalg.norm( aquiStatic.GetPoint("LASI").GetValues()[frameInit:frameEnd,:].mean(axis=0) - aquiStatic.GetPoint("RASI").GetValues()[frameInit:frameEnd,:].mean(axis=0))


        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Pelvis"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)


        # --- Hip Joint centers location
        # anthropometric parameter computed
        if "LeftAsisTrocanterDistance" in self.mp and self.mp["LeftAsisTrocanterDistance"] != 0:
            LOGGER.logger.debug("LeftAsisTrocanterDistance defined from your vsk file")
            self.mp_computed['LeftAsisTrocanterDistance'] = self.mp["LeftAsisTrocanterDistance"]
        else:
            self.mp_computed['LeftAsisTrocanterDistance'] = 0.1288*self.mp['LeftLegLength']-48.56

        if "RightAsisTrocanterDistance" in self.mp and self.mp["RightAsisTrocanterDistance"] != 0:
            LOGGER.logger.debug("RightAsisTrocanterDistance defined from your vsk file")
            self.mp_computed['RightAsisTrocanterDistance'] = self.mp["RightAsisTrocanterDistance"]
        else:
            self.mp_computed['RightAsisTrocanterDistance'] = 0.1288*self.mp['RightLegLength']-48.56

        self.mp_computed['MeanlegLength'] = np.mean( [self.mp['LeftLegLength'],self.mp['RightLegLength'] ])

        # local Position of the hip joint centers

        LHJC_loc,RHJC_loc= modelDecorator.davisRegression(self.mp,self.mp_computed,
                                                    markerDiameter = markerDiameter,
                                                    basePlate = basePlate)


        # left
        if tf.static.isNodeExist("LHJC"):
            nodeLHJC = tf.static.getNode_byLabel("LHJC")

        else:
            tf.static.addNode("LHJC_cgm1",LHJC_loc,positionType="Local",desc = "Davis")
            tf.static.addNode("LHJC",LHJC_loc,positionType="Local",desc = "Davis")
            nodeLHJC = tf.static.getNode_byLabel("LHJC")

        btkTools.smartAppendPoint(aquiStatic,"LHJC",
                    nodeLHJC.m_global* np.ones((pfn,3)),
                    desc=nodeLHJC.m_desc)

        # right
        if tf.static.isNodeExist("RHJC"):
            nodeRHJC = tf.static.getNode_byLabel("RHJC")
        else:
            tf.static.addNode("RHJC_cgm1",RHJC_loc,positionType="Local",desc = "Davis")
            tf.static.addNode("RHJC",RHJC_loc,positionType="Local",desc = "Davis")
            nodeRHJC = tf.static.getNode_byLabel("RHJC")

        btkTools.smartAppendPoint(aquiStatic,"RHJC",
                    nodeRHJC.m_global*np.ones((pfn,3)),
                    desc=nodeRHJC.m_desc)

        val=(aquiStatic.GetPoint("LHJC").GetValues() + aquiStatic.GetPoint("RHJC").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"midHJC",val,desc="")
        seg.addCalibrationMarkerLabel("midHJC")

        #nodes
        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        for label in seg.m_calibration_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # # add lumbar5
        # pelvisScale = np.linalg.norm(nodeLHJC.m_local-nodeRHJC.m_local)
        # offset = (nodeLHJC.m_local+nodeRHJC.m_local)/2.0
        #
        # TopLumbar5 = offset +  (np.array([ 0, 0, 0.925* pelvisScale]))
        # tf.static.addNode("TL5",TopLumbar5,positionType="Local")
        #
        # com = offset + (TopLumbar5-offset)*0.895



        #nodeL5 = tf.static.getNode_byLabel("TL5")
        #btkTools.smartAppendPoint(aquiStatic,"TL5",
        #            nodeL5.m_global*np.ones((pfn,3)),
        #            desc="")



    def _left_thigh_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):

        pfn = aquiStatic.GetPointFrameNumber()

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg = self.getSegment("Left Thigh")
        seg.resetMarkerLabels()

        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#KNE
        pt2=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#HJC
        pt3=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#THI

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Left Thigh"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- knee Joint centers location from chord method
        if "LeftThighRotation" in self.mp and self.mp["LeftThighRotation"] != 0:
            LOGGER.logger.debug("LeftThighRotation defined from your vsk file")
            self.mp_computed["LeftThighRotationOffset"] = self.mp["LeftThighRotation"]
        else:
            self.mp_computed["LeftThighRotationOffset"] = 0.0

        LKJC = modelDecorator.VCMJointCentre( (self.mp["LeftKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=-self.mp_computed["LeftThighRotationOffset"] )

        if tf.static.isNodeExist("LKJC"):
            nodeLKJC = tf.static.getNode_byLabel("LKJC")
        else:
            tf.static.addNode("LKJC_chord",LKJC,positionType="Global",desc = "Chord")
            tf.static.addNode("LKJC",LKJC,positionType="Global",desc = "Chord")
            nodeLKJC = tf.static.getNode_byLabel("LKJC")

        btkTools.smartAppendPoint(aquiStatic,"LKJC",
                    nodeLKJC.m_global* np.ones((pfn,3)),
                    desc=nodeLKJC.m_desc)

        # node for all markers
        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # for label in seg.m_calibration_markers:
        #     globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        #     tf.static.addNode(label,globalPosition,positionType="Global")

        node_prox = self.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC")
        tf.static.addNode("LHJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

        #seg.addTrackingMarkerLabel("LHJC")
        #seg.addCalibrationMarkerLabel("LKJC")


    def _right_thigh_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):

        pfn = aquiStatic.GetPointFrameNumber()

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg = self.getSegment("Right Thigh")
        seg.resetMarkerLabels()

        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right Thigh"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)


        # --- knee Joint centers location
        if "RightThighRotation" in self.mp and self.mp["RightThighRotation"] != 0:
            LOGGER.logger.debug("RightThighRotation defined from your vsk file")
            self.mp_computed["RightThighRotationOffset"] = self.mp["RightThighRotation"]
        else:
            self.mp_computed["RightThighRotationOffset"] = 0.0

        RKJC = modelDecorator.VCMJointCentre( (self.mp["RightKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3,beta=self.mp_computed["RightThighRotationOffset"] ) # could consider a previous offset

        if tf.static.isNodeExist("RKJC"):
            nodeRKJC = tf.static.getNode_byLabel("RKJC")
        else:
            tf.static.addNode("RKJC_chord",RKJC,positionType="Global",desc = "Chord")
            tf.static.addNode("RKJC",RKJC,positionType="Global",desc = "Chord")
            nodeRKJC = tf.static.getNode_byLabel("RKJC")

        btkTools.smartAppendPoint(aquiStatic,"RKJC",
                    nodeRKJC.m_global* np.ones((pfn,3)),
                    desc=nodeRKJC.m_desc)


        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # for label in seg.m_calibration_markers:
        #     globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        #     tf.static.addNode(label,globalPosition,positionType="Global")

        node_prox = self.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC")
        tf.static.addNode("RHJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

        # seg.addTrackingMarkerLabel("RHJC")
        # seg.addCalibrationMarkerLabel("RKJC")

    def _left_shank_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):

        pfn = aquiStatic.GetPointFrameNumber()

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0


        seg = self.getSegment("Left Shank")
        seg.resetMarkerLabels()

        # ---  additional markers and Update of the marker segment list


        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Left Shank"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)


        # --- ankle Joint centers location
        if "LeftShankRotation" in self.mp and self.mp["LeftShankRotation"] != 0:
            LOGGER.logger.debug("LeftShankRotation defined from your vsk file")
            self.mp_computed["LeftShankRotationOffset"] = self.mp["LeftShankRotation"]
        else:
            self.mp_computed["LeftShankRotationOffset"]=0.0

        LAJC = modelDecorator.VCMJointCentre( (self.mp["LeftAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=-self.mp_computed["LeftShankRotationOffset"] )

        # --- node manager
        if tf.static.isNodeExist("LAJC"):
            nodeLAJC = tf.static.getNode_byLabel("LAJC")
        else:
            tf.static.addNode("LAJC_chord",LAJC,positionType="Global",desc = "Chord")
            tf.static.addNode("LAJC",LAJC,positionType="Global",desc = "Chord")
            nodeLAJC = tf.static.getNode_byLabel("LAJC")

        btkTools.smartAppendPoint(aquiStatic,"LAJC",
                    nodeLAJC.m_global* np.ones((pfn,3)),
                    desc=nodeLAJC.m_desc)


        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # for label in seg.m_calibration_markers:
        #     globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        #     tf.static.addNode(label,globalPosition,positionType="Global")

        node_prox = self.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC")
        tf.static.addNode("LKJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

        # seg.addTrackingMarkerLabel("LKJC")
        # seg.addCalibrationMarkerLabel("LAJC")

    def _right_shank_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):

        pfn = aquiStatic.GetPointFrameNumber()

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0



        seg = self.getSegment("Right Shank")
        seg.resetMarkerLabels()


        # --- Construction of the technical Referential
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right Shank"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- ankle Joint centers location
        if "RightShankRotation" in self.mp and self.mp["RightShankRotation"] != 0:
            LOGGER.logger.debug("RightShankRotation defined from your vsk file")
            self.mp_computed["RightShankRotationOffset"] = self.mp["RightShankRotation"]
        else:
            self.mp_computed["RightShankRotationOffset"]=0.0

        RAJC = modelDecorator.VCMJointCentre( (self.mp["RightAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["RightShankRotationOffset"] )

        # --- node manager
        if tf.static.isNodeExist("RAJC"):
            nodeRAJC = tf.static.getNode_byLabel("RAJC")
        else:
            tf.static.addNode("RAJC_chord",RAJC,positionType="Global",desc = "Chord")
            tf.static.addNode("RAJC",RAJC,positionType="Global",desc = "Chord")
            nodeRAJC = tf.static.getNode_byLabel("RAJC")

        btkTools.smartAppendPoint(aquiStatic,"RAJC",
                    nodeRAJC.m_global* np.ones((pfn,3)),
                    desc=nodeRAJC.m_desc)


        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # for label in seg.m_calibration_markers:
        #     globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        #     tf.static.addNode(label,globalPosition,positionType="Global")

        node_prox = self.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC")
        tf.static.addNode("RKJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

        # seg.addTrackingMarkerLabel("RKJC")
        # seg.addCalibrationMarkerLabel("RAJC")

    def _left_unCorrectedFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):

        seg = self.getSegment("Left Foot")
        #seg.resetMarkerLabels()

        # ---  additional markers and Update of the marker segment list
        #seg.addMarkerLabel("LKJC") !!!

        if "useBodyBuilderFoot" in options.keys() and options["useBodyBuilderFoot"]:
            LOGGER.logger.debug("You use a Left uncorrected foot sequence different than native CGM1")
            dictRef["Left Foot"]={"TF" : {'sequence':"ZYX", 'labels':   ["LTOE","LAJC","LKJC","LAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis


        # --- Construction of the technical Referential
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#LTOE
        pt2=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#AJC

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        if dictRef["Left Foot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)

            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))
            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))
            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Left Foot"]["TF"]['sequence'])

        else:
            distalShank = self.getSegment("Left Shank")
            proximalShank = self.getSegment("Left Shank Proximal")

            # uncorrected Refrence with dist shank
            v=distalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))
            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))
            x_dist,y_dist,z_dist,R_dist=frame.setFrameData(a1,a2,dictRef["Left Foot"]["TF"]['sequence'])

            # uncorrected Refrence with prox shank
            v=proximalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))
            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))
            x_prox,y_prox,z_prox,R_prox=frame.setFrameData(a1,a2,dictRef["Left Foot"]["TF"]['sequence'])

            self._R_leftUnCorrfoot_dist_prox = np.dot(R_prox.T,R_dist) # will be used for placing the foot uncorrected RF

            if "viconCGM1compatible" in options.keys() and options["viconCGM1compatible"]:
                if distalShank.getReferential("TF").static.getNode_byLabel("LAJC").m_desc != "mid":
                    x,y,z,R = x_prox,y_prox,z_prox,R_prox
                else:
                    x,y,z,R = x_dist,y_dist,z_dist,R_dist
            else:
                x,y,z,R = x_dist,y_dist,z_dist,R_dist

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # for label in seg.m_calibration_markers:
        #     globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        #     tf.static.addNode(label,globalPosition,positionType="Global")

        node_prox = self.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC")
        tf.static.addNode("LAJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

        # seg.addTrackingMarkerLabel("LAJC") # for LS fitting




    def _right_unCorrectedFoot_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):

        seg = self.getSegment("Right Foot")
        seg.resetMarkerLabels()

        # ---  additional markers and Update of the marker segment list
        #seg.addMarkerLabel("RKJC")


        if "useBodyBuilderFoot" in options.keys() and options["useBodyBuilderFoot"]:
            LOGGER.logger.debug("You use a right uncorrected foot sequence different than native CGM1")
            dictRef["Right Foot"]={"TF" : {'sequence':"ZYX", 'labels':   ["RTOE","RAJC","RKJC","RAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis



        # --- Construction of the anatomical Referential
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        if dictRef["Right Foot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)

            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))
            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))
            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right Foot"]["TF"]['sequence'])

        else:
            distalShank = self.getSegment("Right Shank")
            proximalShank = self.getSegment("Right Shank Proximal")

            # uncorrected Refrence with dist shank
            v=distalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))
            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))
            x_dist,y_dist,z_dist,R_dist=frame.setFrameData(a1,a2,dictRef["Right Foot"]["TF"]['sequence'])

            # uncorrected Refrence with prox shank
            v=proximalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))
            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))
            x_prox,y_prox,z_prox,R_prox=frame.setFrameData(a1,a2,dictRef["Right Foot"]["TF"]['sequence'])

            self._R_rightUnCorrfoot_dist_prox = np.dot(R_prox.T,R_dist)

            if "viconCGM1compatible" in options.keys() and options["viconCGM1compatible"]:
                if distalShank.getReferential("TF").static.getNode_byLabel("RAJC").m_desc != "mid":
                    x,y,z,R = x_prox,y_prox,z_prox,R_prox
                else:
                    x,y,z,R = x_dist,y_dist,z_dist,R_dist
            else:
                x,y,z,R = x_dist,y_dist,z_dist,R_dist

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- node manager
        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # for label in seg.m_calibration_markers:
        #     globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        #     tf.static.addNode(label,globalPosition,positionType="Global")

        node_prox = self.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC")
        tf.static.addNode("RAJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)
        # seg.addTrackingMarkerLabel("RAJC")


    def _pelvis_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):

        seg=self.getSegment("Pelvis")


        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Pelvis"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Pelvis"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Pelvis"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Pelvis"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Pelvis"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # --- node manager
        for node in seg.getReferential("TF").static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

        # length
        lhjc = seg.anatomicalFrame.static.getNode_byLabel("LHJC").m_local
        rhjc = seg.anatomicalFrame.static.getNode_byLabel("RHJC").m_local
        seg.setLength(np.linalg.norm(lhjc-rhjc))

        pelvisScale = np.linalg.norm(lhjc-rhjc)
        offset = (lhjc+rhjc)/2.0

        TopLumbar5 = offset +  (np.array([ 0, 0, 0.925]))* pelvisScale
        seg.anatomicalFrame.static.addNode("TL5",TopLumbar5,positionType="Local")

        com = offset + (TopLumbar5-offset)*0.895
        seg.anatomicalFrame.static.addNode("com",com,positionType="Local")

    def _left_thigh_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):

        seg=self.getSegment("Left Thigh")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Left Thigh"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Left Thigh"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Left Thigh"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Left Thigh"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Left Thigh"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)


        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))


        # --- node manager
        for node in seg.getReferential("TF").static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

        # --- compute length
        hjc = seg.anatomicalFrame.static.getNode_byLabel("LHJC").m_local
        kjc = seg.anatomicalFrame.static.getNode_byLabel("LKJC").m_local

        seg.setLength(np.linalg.norm(kjc-hjc))


    def _right_thigh_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):

        seg=self.getSegment("Right Thigh")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Right Thigh"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # --- node manager
        for node in seg.getReferential("TF").static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

        # --- compute lenght
        hjc = seg.anatomicalFrame.static.getNode_byLabel("RHJC").m_local
        kjc = seg.anatomicalFrame.static.getNode_byLabel("RKJC").m_local

        seg.setLength(np.linalg.norm(kjc-hjc))

    def _left_shank_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):

        seg=self.getSegment("Left Shank")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Left Shank"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Left Shank"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Left Shank"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Left Shank"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Left Shank"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # --- node manager
        for node in seg.getReferential("TF").static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

        # --- compute length
        kjc = seg.anatomicalFrame.static.getNode_byLabel("LKJC").m_local
        ajc = seg.anatomicalFrame.static.getNode_byLabel("LAJC").m_local

        seg.setLength(np.linalg.norm(ajc-kjc))

    def _left_shankProximal_AnatomicalCalibrate(self,aquiStatic,dictAnat,frameInit,frameEnd,options=None):

        if self.m_useLeftTibialTorsion:
            tibialTorsion = -1.0*np.deg2rad(self.mp_computed["LeftTibialTorsionOffset"])
        else:
            tibialTorsion = 0.0


        seg=self.getSegment("Left Shank Proximal")


        # --- set static anatomical Referential
        # Rotation of the static anatomical Referential by the tibial Torsion angle
        rotZ_tibRot = np.eye(3,3)
        rotZ_tibRot[0,0] = np.cos(tibialTorsion)
        rotZ_tibRot[0,1] = np.sin(tibialTorsion)
        rotZ_tibRot[1,0] = - np.sin(tibialTorsion)
        rotZ_tibRot[1,1] = np.cos(tibialTorsion)

        R = np.dot(seg.anatomicalFrame.static.getRotation(),rotZ_tibRot)

        # update frame
        csFrame=frame.Frame()
        csFrame.update(R,seg.anatomicalFrame.static.getTranslation())
        seg.anatomicalFrame.setStaticFrame(csFrame)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # --- node manager
        for node in tf.static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())


    def _right_shank_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):

        seg=self.getSegment("Right Shank")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Shank"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Shank"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Shank"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Shank"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Right Shank"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # --- node manager
        for node in tf.static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())


        # --- compute length
        kjc = seg.anatomicalFrame.static.getNode_byLabel("RKJC").m_local
        ajc = seg.anatomicalFrame.static.getNode_byLabel("RAJC").m_local
        seg.setLength(np.linalg.norm(ajc-kjc))


    def _right_shankProximal_AnatomicalCalibrate(self,aquiStatic,dictAnat,frameInit,frameEnd,options=None):

        if self.m_useRightTibialTorsion:
            tibialTorsion = np.deg2rad(self.mp_computed["RightTibialTorsionOffset"])
        else:
            tibialTorsion = 0.0


        seg=self.getSegment("Right Shank Proximal")

        # --- set static anatomical Referential
        # Rotation of the static anatomical Referential by the tibial Torsion angle
        rotZ_tibRot = np.eye(3,3)
        rotZ_tibRot[0,0] = np.cos(tibialTorsion)
        rotZ_tibRot[0,1] = np.sin(tibialTorsion)
        rotZ_tibRot[1,0] = - np.sin(tibialTorsion)
        rotZ_tibRot[1,1] = np.cos(tibialTorsion)

        R = np.dot(seg.anatomicalFrame.static.getRotation(),rotZ_tibRot)

        csFrame=frame.Frame()
        csFrame.update(R,seg.anatomicalFrame.static.getTranslation() )
        seg.anatomicalFrame.setStaticFrame(csFrame)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # node manager
        # --- node manager
        for node in tf.static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())


    def _left_foot_corrected_calibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd,options = None):

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0


        seg=self.getSegment("Left Foot")

        if "useBodyBuilderFoot" in options.keys() and options["useBodyBuilderFoot"]:
            LOGGER.logger.debug("You use a Left corrected foot sequence different than native CGM1")
            dictAnatomic["Left Foot"]={'sequence':"ZYX", 'labels':  ["LTOE","LHEE","LKJC","LAJC"]}    # corrected foot


        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # LTOE
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # LHEE

        if ("leftFlatFoot" in options.keys() and options["leftFlatFoot"]):
            LOGGER.logger.debug ("option (leftFlatFoot) enable")
            if ("LeftSoleDelta" in self.mp.keys() and self.mp["LeftSoleDelta"]!=0):
                LOGGER.logger.debug ("option (LeftSoleDelta) compensation")

            pt2[2] = pt1[2]+self.mp['LeftSoleDelta']


        if dictAnatomic["Left Foot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)
        else:
            distalShank = self.getSegment("Left Shank")
            proximalShank = self.getSegment("Left Shank Proximal")
            if "viconCGM1compatible" in options.keys() and options["viconCGM1compatible"]:
                if distalShank.getReferential("TF").static.getNode_byLabel("LAJC").m_desc != "mid":
                    v=proximalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
                else:
                    v=distalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
            else:
                v=distalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Left Foot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Left Foot"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        # This section compute the actual Relative Rotation between anatomical and technical Referential
        trueRelativeMatrixAnatomic = np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation())
        y,x,z = euler.euler_yxz(trueRelativeMatrixAnatomic)

        # the native CGM relative rotation leaves out the rotation around Z
        rotX =np.array([[1,0,0],
                        [0,np.cos(x),-np.sin(x)],
                         [0,np.sin(x),np.cos(x)]])
        rotY =np.array([[np.cos(y),0,np.sin(y)],
                        [0,1,0],
                         [-np.sin(y),0,np.cos(y)]])

        relativeMatrixAnatomic = np.dot(rotY,rotX)

        tf.setRelativeMatrixAnatomic( relativeMatrixAnatomic)
        tf.additionalInfos["trueRelativeMatrix"] = trueRelativeMatrixAnatomic

        # --- node manager
        for node in tf.static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())


        # --- compute amthropo
        # length
        toe = seg.anatomicalFrame.static.getNode_byLabel("LTOE").m_local
        hee = seg.anatomicalFrame.static.getNode_byLabel("LHEE").m_local
        seg.setLength(np.linalg.norm(toe-hee)- markerDiameter/2.0)

        # com
        toe = seg.anatomicalFrame.static.getNode_byLabel("LTOE").m_global
        hee = seg.anatomicalFrame.static.getNode_byLabel("LHEE").m_global
        footLongAxis = (toe-hee)/np.linalg.norm(toe-hee)

        com = hee + 0.5 * seg.m_bsp["length"] * footLongAxis

        seg.anatomicalFrame.static.addNode("com",com,positionType="Global")


        # foot origin offset and Toe origin
        local_oo = np.array([-11, -11, -120])/169.0*seg.m_bsp["length"]
        local_to =local_oo + np.array([0, 0, -seg.m_bsp["length"]/3.0])

        seg.anatomicalFrame.static.addNode("FootOriginOffset",local_oo,positionType="Local")
        seg.anatomicalFrame.static.addNode("ToeOrigin",local_to,positionType="Local")

    def _right_foot_corrected_calibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd,options = None):


        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment("Right Foot")

        if "useBodyBuilderFoot" in options.keys() and options["useBodyBuilderFoot"]:
            LOGGER.logger.debug("You use a Right corrected foot sequence different than native CGM1")
            dictAnatomic["Right Foot"]={'sequence':"ZYX", 'labels':  ["RTOE","RHEE","RKJC","RAJC"]}    # corrected foot

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        #pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        if ("rightFlatFoot" in options.keys() and options["rightFlatFoot"]):
            LOGGER.logger.debug ("option (rightFlatFoot) enable")

            if ("RightSoleDelta" in self.mp.keys() and self.mp["RightSoleDelta"]!=0):
                LOGGER.logger.debug ("option (RightSoleDelta) compensation")

            pt2[2] = pt1[2]+self.mp['RightSoleDelta']


        if dictAnatomic["Right Foot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)
        else:
            distalShank = self.getSegment("Right Shank")
            proximalShank = self.getSegment("Right Shank Proximal")
            if "viconCGM1compatible" in options.keys() and options["viconCGM1compatible"]:
                if distalShank.getReferential("TF").static.getNode_byLabel("RAJC").m_desc != "mid":
                    v=proximalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
                else:
                    v=distalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
            else:
                v=distalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)


        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Right Foot"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        # actual Relative Rotation
        trueRelativeMatrixAnatomic = np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation())
        y,x,z = euler.euler_yxz(trueRelativeMatrixAnatomic)

        # native CGM relative rotation
        rotX =np.array([[1,0,0],
                        [0,np.cos(x),-np.sin(x)],
                         [0,np.sin(x),np.cos(x)]])

        rotY =np.array([[np.cos(y),0,np.sin(y)],
                        [0,1,0],
                         [-np.sin(y),0,np.cos(y)]])

        relativeMatrixAnatomic = np.dot(rotY,rotX)

        tf.setRelativeMatrixAnatomic(relativeMatrixAnatomic)

        # --- node manager
        for node in seg.getReferential("TF").static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

        # --- anthropo
        # length
        toe = seg.anatomicalFrame.static.getNode_byLabel("RTOE").m_local
        hee = seg.anatomicalFrame.static.getNode_byLabel("RHEE").m_local
        seg.setLength(np.linalg.norm(toe-hee)- markerDiameter/2.0)

        # com
        toe = seg.anatomicalFrame.static.getNode_byLabel("RTOE").m_global
        hee = seg.anatomicalFrame.static.getNode_byLabel("RHEE").m_global
        com = (toe+hee)/2.0

        seg.anatomicalFrame.static.addNode("com",com,positionType="Global")


        # foot origin offset and Toe origin
        local_oo = np.array([-11, 11, -120])/169.0*seg.m_bsp["length"]
        local_to =local_oo + np.array([0, 0, -seg.m_bsp["length"]/3.0])

        seg.anatomicalFrame.static.addNode("FootOriginOffset",local_oo,positionType="Local")
        seg.anatomicalFrame.static.addNode("ToeOrigin",local_to,positionType="Local")


    def _rotateAnatomicalFrame(self,segmentLabel, angle, aquiStatic, dictAnatomic,frameInit,frameEnd,):

        seg=self.getSegment(segmentLabel)

        angle = np.deg2rad(angle)

        # --- set static anatomical Referential
        # Rotation of the static anatomical Referential by the tibial Torsion angle
        rotZ = np.eye(3,3)
        rotZ[0,0] = np.cos(angle)
        rotZ[0,1] = -np.sin(angle)
        rotZ[1,0] = np.sin(angle)
        rotZ[1,1] = np.cos(angle)

        R = np.dot(seg.anatomicalFrame.static.getRotation(),rotZ) # apply rotation


        csFrame=frame.Frame() #  WARNING Creation of a new Frame remove all former node
        csFrame.update(R,seg.anatomicalFrame.static.getTranslation() )
        seg.anatomicalFrame.setStaticFrame(csFrame)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))


        # add node
        previous_nodes = tf.static.getNodes() # get nodes from technical frames

        for node in previous_nodes:
            globalPosition=node.getGlobal()
            seg.anatomicalFrame.static.addNode(node.getLabel(),globalPosition,positionType="Global",desc = node.getDescription())


    # ---- Offsets -------

    def getThighOffset(self,side= "both"):
        """
        return the thigh offset. Angle between the projection of the lateral thigh marker and the knee flexion axis

        Args:
            side (string, Optional[both]): lower limb side (both, left or right)
        """

        if side == "both" or side=="left":

            thi = self.getSegment("Left Thigh").anatomicalFrame.static.getNode_byLabel("LTHI").m_global - self.getSegment("Left Thigh").anatomicalFrame.static.getNode_byLabel("LKJC").m_global
            angle = np.rad2deg(np.arctan2( -1.0*np.dot(thi, self.getSegment("Left Thigh").anatomicalFrame.static.m_axisX),
                                    np.dot(thi, self.getSegment("Left Thigh").anatomicalFrame.static.m_axisY)))
            self.mp_computed["LeftThighRotationOffset"]= angle # angle needed : Thi toward knee flexion

        if side == "both" or side=="right":

            thi = self.getSegment("Right Thigh").anatomicalFrame.static.getNode_byLabel("RTHI").m_global - \
                  self.getSegment("Right Thigh").anatomicalFrame.static.getNode_byLabel("RKJC").m_global

            angle = np.rad2deg(np.arctan2( -1.0*np.dot(thi, self.getSegment("Right Thigh").anatomicalFrame.static.m_axisX),
                                    -1.0*np.dot(thi, self.getSegment("Right Thigh").anatomicalFrame.static.m_axisY)))

            self.mp_computed["RightThighRotationOffset"] = angle # angle needed : Thi toward knee flexion
            LOGGER.logger.debug(" right Thigh Offset => %s " % str(self.mp_computed["RightThighRotationOffset"]))



    def getShankOffsets(self, side = "both"):
        """
        return  the shank offset, ie the angle between the projection of the lateral shank marker and the ankle flexion axis

        Args:
            side (string, Optional[both]): lower limb side (both, left or right)

        """

        if side == "both" or side == "left" :

            tib = self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LTIB").m_global - \
                  self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LAJC").m_global
            angle = np.rad2deg(np.arctan2( -1.0*np.dot(tib, self.getSegment("Left Shank").anatomicalFrame.static.m_axisX),
                                    np.dot(tib, self.getSegment("Left Shank").anatomicalFrame.static.m_axisY)))
            self.mp_computed["LeftShankRotationOffset"]= angle


        if side == "both" or side == "right" :

            tib = self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RTIB").m_global - \
                  self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RAJC").m_global

            angle = np.rad2deg(np.arctan2( -1.0*np.dot(tib, self.getSegment("Right Shank").anatomicalFrame.static.m_axisX),
                                    -1.0*np.dot(tib, self.getSegment("Right Shank").anatomicalFrame.static.m_axisY)))

            self.mp_computed["RightShankRotationOffset"]= angle


    def getTibialTorsionOffset(self, side = "both"):
        """
        return the tibial torsion offsets :

        Args:
            side (string, Optional[both]): lower limb side (both, left or right)
        """

        if side == "both" or side == "left" :

            kneeFlexionAxis=    self.getSegment("Left Thigh").anatomicalFrame.static.m_axisY

            ankleFlexionAxis=    self.getSegment("Left Shank").anatomicalFrame.static.m_axisY


            angle = np.rad2deg(np.arctan2( np.dot(kneeFlexionAxis, self.getSegment("Left Shank").anatomicalFrame.static.m_axisY),
                                            np.dot(kneeFlexionAxis, self.getSegment("Left Shank").anatomicalFrame.static.m_axisX))- \
                                np.arctan2( np.dot(ankleFlexionAxis, self.getSegment("Left Shank").anatomicalFrame.static.m_axisY),
                                            np.dot(ankleFlexionAxis, self.getSegment("Left Shank").anatomicalFrame.static.m_axisX)))

            self.mp_computed["LeftTibialTorsionOffset"] = angle


        if side == "both" or side == "right" :


            kneeFlexionAxis=    self.getSegment("Right Thigh").anatomicalFrame.static.m_axisY

            ankleFlexionAxis=    self.getSegment("Right Shank").anatomicalFrame.static.m_axisY


            angle = np.rad2deg(-1.0*np.arctan2( np.dot(kneeFlexionAxis, self.getSegment("Right Shank").anatomicalFrame.static.m_axisY),
                                            np.dot(kneeFlexionAxis, self.getSegment("Right Shank").anatomicalFrame.static.m_axisX))+ \
                                np.arctan2( np.dot(ankleFlexionAxis, self.getSegment("Right Shank").anatomicalFrame.static.m_axisY),
                                            np.dot(ankleFlexionAxis, self.getSegment("Right Shank").anatomicalFrame.static.m_axisX)))

            self.mp_computed["RightTibialTorsionOffset"] = angle
            LOGGER.logger.debug(" Right tibial torsion => %s " % str(self.mp_computed["RightTibialTorsionOffset"]))

    def getAbdAddAnkleJointOffset(self,side="both"):
        """
        return the  Abd/Add ankle offset, ie angle in the frontal plan between the ankle marker and the ankle flexion axis

        Args:
            side (string, Optional[both]): lower limb side (both, left or right)
        """
        if side == "both" or side == "left" :

            AnkleAxis = self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LANK").m_global - \
                  self.getSegment("Left Shank").anatomicalFrame.static.getNode_byLabel("LAJC").m_global

            angle = np.rad2deg(np.arctan2( -1.0*np.dot(self.getSegment("Left Shank").anatomicalFrame.static.m_axisZ,  AnkleAxis ),
                                                np.dot(self.getSegment("Left Shank").anatomicalFrame.static.m_axisY,  AnkleAxis )))

            self.mp_computed["LeftAnkleAbAddOffset"] = angle

            LOGGER.logger.debug(" LeftAnkleAbAddOffset => %s " % str(self.mp_computed["LeftAnkleAbAddOffset"]))

        if side == "both" or side == "right" :

            AnkleAxis = self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RANK").m_global - \
                  self.getSegment("Right Shank").anatomicalFrame.static.getNode_byLabel("RAJC").m_global

            angle = np.rad2deg(np.arctan2( -1.0*np.dot(self.getSegment("Right Shank").anatomicalFrame.static.m_axisZ,  AnkleAxis ),
                                    -1.0*np.dot(self.getSegment("Right Shank").anatomicalFrame.static.m_axisY,  AnkleAxis )))

            self.mp_computed["RightAnkleAbAddOffset"] = angle

            LOGGER.logger.debug(" RightAnkleAbAddOffset => %s " % str(self.mp_computed["RightAnkleAbAddOffset"]))


    def getFootOffset(self, side = "both"):
        """
        return the foot offsets, ie the plantar flexion offset and the rotation offset

        Args:
            side (string, Optional[both]): lower limb side (both, left or right)
        """


        if side == "both" or side == "left" :
            R = self.getSegment("Left Foot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = euler.euler_yxz(R)

            self.mp_computed["LeftStaticPlantFlexOffset"] = -1.0*np.rad2deg(y)
            LOGGER.logger.debug(" LeftStaticPlantFlexOffset => %s " % str(self.mp_computed["LeftStaticPlantFlexOffset"]))

            self.mp_computed["LeftStaticRotOffset"] = -1.0*np.rad2deg(x)
            LOGGER.logger.debug(" LeftStaticRotOffset => %s " % str(self.mp_computed["LeftStaticRotOffset"]))


        if side == "both" or side == "right" :
            R = self.getSegment("Right Foot").getReferential("TF").relativeMatrixAnatomic
            y,x,z = euler.euler_yxz(R)

            self.mp_computed["RightStaticPlantFlexOffset"] = -1.0*np.rad2deg(y)
            LOGGER.logger.debug(" RightStaticPlantFlexOffset => %s " % str(self.mp_computed["RightStaticPlantFlexOffset"]))

            self.mp_computed["RightStaticRotOffset"] = np.rad2deg(x)
            LOGGER.logger.debug(" RightStaticRotOffset => %s " % str(self.mp_computed["RightStaticRotOffset"]))

    # ----- Motion --------------
    def computeOptimizedSegmentMotion(self,aqui,segments, dictRef,dictAnat,motionMethod,options ):
        """Compute poses of both **Technical and Anatomical** coordinate systems
        for specific segments of the model

        Args:
            aqui (btk.Acquisition): motion acquisitiuon
            segments (list): segments of the model
            dictRef (dict): technical referential definitions
            dictAnat (dict): anatomical referential definitions
            motionMethod (enums.motionMethod): segmental motion method to apply
            options (dict): passed known-arguments

        """


        # ---remove all  direction marker from tracking markers.
        if self.staExpert:
            for seg in self.m_segmentCollection:
                selectedTrackingMarkers=list()
                for marker in seg.m_tracking_markers:
                    if marker in self.__class__.TRACKING_MARKERS : # get class variable MARKER even from child
                        selectedTrackingMarkers.append(marker)
                seg.m_tracking_markers= selectedTrackingMarkers


        LOGGER.logger.debug("--- Segmental Least-square motion process ---")
        if "Pelvis" in segments:
            self._pelvis_motion_optimize(aqui, dictRef, motionMethod)
            self._anatomical_motion(aqui,"Pelvis",originLabel = str(dictAnat["Pelvis"]['labels'][3]))

        if "Left Thigh" in segments:
            self._left_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Thigh",originLabel = "LKJC")


        if "Right Thigh" in segments:
            self._right_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Thigh",originLabel = "RKJC")


        if "Left Shank" in segments:
            self._left_shank_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Shank",originLabel = "LAJC")

        if "Right Shank" in segments:
            self._right_shank_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Shank",originLabel = "RAJC")

        if "Left Foot" in segments:
            self._left_foot_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Foot",originLabel = "LHEE")

        if "Right Foot" in segments:
            self._right_foot_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Foot",originLabel = "RHEE")





    def computeMotion(self,aqui, dictRef,dictAnat, motionMethod,options=None ):
        """
        Compute poses of both **Technical and Anatomical** coordinate systems

        Args:
            aqui (btk.acquisition): acquisition
            dictRef (dict): technical referential definitions
            dictAnat (dict): anatomical referential definitions
            motionMethod (enums.motionMethod): segmental motion method
            options (dict,optional[None]): passed arguments to embedded functions

        options:
            * pigStatic (bool) : compute foot cordinate system according the Vicon Plugin-gait
            * forceFoot6DoF (bool): apply 6DOF pose optimisation on the foot
        """

        LOGGER.logger.debug("=====================================================")
        LOGGER.logger.debug("===================  CGM MOTION   ===================")
        LOGGER.logger.debug("=====================================================")

        pigStaticProcessing= True if "pigStatic" in options.keys() and options["pigStatic"] else False
        forceFoot6DoF= True if "forceFoot6DoF" in options.keys() and options["forceFoot6DoF"] else False


        if motionMethod == enums.motionMethod.Determinist: #cmf.motionMethod.Native:

            #if not pigStaticProcessing:
            LOGGER.logger.debug(" - Pelvis - motion -")
            LOGGER.logger.debug(" -------------------")
            self._pelvis_motion(aqui, dictRef, dictAnat)

            LOGGER.logger.debug(" - Left Thigh - motion -")
            LOGGER.logger.debug(" -----------------------")
            self._left_thigh_motion(aqui, dictRef, dictAnat,options=options)


            # if rotation offset from knee functional calibration methods
            if self.mp_computed["LeftKneeFuncCalibrationOffset"]:
                offset = self.mp_computed["LeftKneeFuncCalibrationOffset"]
                self._rotate_anatomical_motion("Left Thigh",offset,
                                        aqui,options=options)

            LOGGER.logger.debug(" - Right Thigh - motion -")
            LOGGER.logger.debug(" ------------------------")
            self._right_thigh_motion(aqui, dictRef, dictAnat,options=options)


            if  self.mp_computed["RightKneeFuncCalibrationOffset"]:
                offset = self.mp_computed["RightKneeFuncCalibrationOffset"]
                self._rotate_anatomical_motion("Right Thigh",offset,
                                        aqui,options=options)


            LOGGER.logger.debug(" - Left Shank - motion -")
            LOGGER.logger.debug(" -----------------------")
            self._left_shank_motion(aqui, dictRef, dictAnat,options=options)


            LOGGER.logger.debug(" - Left Shank-proximal - motion -")
            LOGGER.logger.debug(" --------------------------------")
            self._left_shankProximal_motion(aqui,dictAnat,options=options)

            LOGGER.logger.debug(" - Right Shank - motion -")
            LOGGER.logger.debug(" ------------------------")
            self._right_shank_motion(aqui, dictRef, dictAnat,options=options)

            LOGGER.logger.debug(" - Right Shank-proximal - motion -")
            LOGGER.logger.debug(" ---------------------------------")
            self._right_shankProximal_motion(aqui,dictAnat,options=options)

            LOGGER.logger.debug(" - Left foot - motion -")
            LOGGER.logger.debug(" ----------------------")

            if pigStaticProcessing:
                self._left_foot_motion_static(aqui, dictAnat,options=options)
            else:
                self._left_foot_motion(aqui, dictRef, dictAnat,options=options)

            LOGGER.logger.debug(" - Right foot - motion -")
            LOGGER.logger.debug(" ----------------------")


            if pigStaticProcessing:
                self._right_foot_motion_static(aqui, dictAnat,options=options)
            else:
                self._right_foot_motion(aqui, dictRef, dictAnat,options=options)


            self._thorax_motion(aqui, dictRef,dictAnat,options=options)
            self._head_motion(aqui, dictRef,dictAnat,options=options)

            self._clavicle_motion("Left",aqui, dictRef,dictAnat,options=options)
            self._constructArmVirtualMarkers("Left", aqui)
            self._upperArm_motion("Left",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Technical")
            self._foreArm_motion("Left",aqui, dictRef,dictAnat,options=options, frameReconstruction="Technical")
            self._upperArm_motion("Left",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Anatomical")
            self._foreArm_motion("Left",aqui, dictRef,dictAnat,options=options, frameReconstruction="Anatomical")
            self._hand_motion("Left",aqui, dictRef,dictAnat,options=options)

            self._clavicle_motion("Right",aqui, dictRef,dictAnat,options=options)
            self._constructArmVirtualMarkers("Right", aqui)
            self._upperArm_motion("Right",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Technical")
            self._foreArm_motion("Right",aqui, dictRef,dictAnat,options=options, frameReconstruction="Technical")
            self._upperArm_motion("Right",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Anatomical")
            self._foreArm_motion("Right",aqui, dictRef,dictAnat,options=options, frameReconstruction="Anatomical")
            self._hand_motion("Right",aqui, dictRef,dictAnat,options=options)


        if motionMethod == enums.motionMethod.Sodervisk:

            # ---remove all  direction marker from tracking markers.
            if self.staExpert:
                for seg in self.m_segmentCollection:
                    selectedTrackingMarkers=list()
                    for marker in seg.m_tracking_markers:
                        if marker in self.__class__.TRACKING_MARKERS :
                            selectedTrackingMarkers.append(marker)
                    seg.m_tracking_markers= selectedTrackingMarkers


            LOGGER.logger.debug("--- Segmental Least-square motion process ---")
            self._pelvis_motion_optimize(aqui, dictRef, motionMethod)
            self._anatomical_motion(aqui,"Pelvis",originLabel = str(dictAnat["Pelvis"]['labels'][3]))


            TopLumbar5=np.zeros((aqui.GetPointFrameNumber(),3))

            for i in range(0,aqui.GetPointFrameNumber()):
                lhjc = aqui.GetPoint("LHJC").GetValues()[i,:]
                rhjc =  aqui.GetPoint("RHJC").GetValues()[i,:]
                pelvisScale = np.linalg.norm(lhjc-rhjc)
                offset = (lhjc+rhjc)/2.0
                R = self.getSegment("Pelvis").anatomicalFrame.motion[i].getRotation()
                TopLumbar5[i,:] = offset +  np.dot(R,(np.array([ 0, 0, 0.925]))* pelvisScale)


            self._TopLumbar5 = TopLumbar5

            self._left_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Thigh",originLabel = str(dictAnat["Left Thigh"]['labels'][3]))

            self._right_thigh_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Thigh",originLabel = str(dictAnat["Right Thigh"]['labels'][3]))


            self._left_shank_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Shank",originLabel = str(dictAnat["Left Shank"]['labels'][3]))
            self._left_shankProximal_motion(aqui,dictAnat,options=options)

            self._right_shank_motion_optimize(aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Shank",originLabel = str(dictAnat["Right Shank"]['labels'][3]))
            self._right_shankProximal_motion(aqui,dictAnat,options=options)

            if forceFoot6DoF:
            # foot
            # issue with least-square optimization :  AJC - HEE and TOE may be inline -> singularities !!
                self._leftFoot_motion_optimize(aqui, dictRef,dictAnat, motionMethod)
                self._anatomical_motion(aqui,"Left Foot",originLabel = str(dictAnat["Left Foot"]['labels'][3]))
                self._rightFoot_motion_optimize(aqui, dictRef,dictAnat, motionMethod)
                self._anatomical_motion(aqui,"Right Foot",originLabel = str(dictAnat["Right Foot"]['labels'][3]))
            else:
                self._left_foot_motion(aqui, dictRef, dictAnat,options=options)
                self._right_foot_motion(aqui, dictRef, dictAnat,options=options)

            self._thorax_motion(aqui, dictRef,dictAnat,options=options)
            self._head_motion(aqui, dictRef,dictAnat,options=options)

            self._clavicle_motion("Left",aqui, dictRef,dictAnat,options=options)
            self._constructArmVirtualMarkers("Left", aqui)
            self._upperArm_motion("Left",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Technical")
            self._foreArm_motion("Left",aqui, dictRef,dictAnat,options=options, frameReconstruction="Technical")
            self._upperArm_motion("Left",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Anatomical")
            self._foreArm_motion("Left",aqui, dictRef,dictAnat,options=options, frameReconstruction="Anatomical")
            self._hand_motion("Left",aqui, dictRef,dictAnat,options=options)

            self._clavicle_motion("Right",aqui, dictRef,dictAnat,options=options)
            self._constructArmVirtualMarkers("Right", aqui)
            self._upperArm_motion("Right",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Technical")
            self._foreArm_motion("Right",aqui, dictRef,dictAnat,options=options, frameReconstruction="Technical")
            self._upperArm_motion("Right",aqui, dictRef,dictAnat,options=options,   frameReconstruction="Anatomical")
            self._foreArm_motion("Right",aqui, dictRef,dictAnat,options=options, frameReconstruction="Anatomical")
            self._hand_motion("Right",aqui, dictRef,dictAnat,options=options)

    def _pelvis_motion(self,aqui, dictRef,dictAnat):

        seg=self.getSegment("Pelvis")

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]        # reinit Technical Frame Motion (USEFUL if you work with several aquisitions)

        #  additional markers
        val=(aqui.GetPoint("LPSI").GetValues() + aqui.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,"SACR",val, desc="")

        val=(aqui.GetPoint("LASI").GetValues() + aqui.GetPoint("RASI").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,"midASIS",val, desc="")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][3])).GetValues()[i,:]

            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))


            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Pelvis"]["TF"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))


        # --- HJCs
        desc_L = seg.getReferential('TF').static.getNode_byLabel("LHJC").m_desc
        desc_R = seg.getReferential('TF').static.getNode_byLabel("RHJC").m_desc

        values_LHJCnode=seg.getReferential('TF').getNodeTrajectory("LHJC")
        values_RHJCnode=seg.getReferential('TF').getNodeTrajectory("RHJC")

        for i in range(0,aqui.GetPointFrameNumber()):
            if not validFrames[i]:
                values_LHJCnode[i,:] = np.zeros(3)
                values_RHJCnode[i,:] = np.zeros(3)


        btkTools.smartAppendPoint(aqui,"LHJC",values_LHJCnode, desc=desc_L)
        btkTools.smartAppendPoint(aqui,"RHJC",values_RHJCnode, desc=desc_R)


        # --- motion of the anatomical referential

        seg.anatomicalFrame.motion=[]
        TopLumbar5=np.zeros((aqui.GetPointFrameNumber(),3))

        # additional markers
        val=(aqui.GetPoint("LHJC").GetValues() + aqui.GetPoint("RHJC").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,"midHJC",val,desc="")

        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictAnat["Pelvis"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))

            # length
            lhjc = aqui.GetPoint("LHJC").GetValues()[i,:]
            rhjc =  aqui.GetPoint("RHJC").GetValues()[i,:]
            pelvisScale = np.linalg.norm(lhjc-rhjc)
            offset = (lhjc+rhjc)/2.0

            TopLumbar5[i,:] = offset +  np.dot(R,(np.array([ 0, 0, 0.925]))* pelvisScale)
            #seg.anatomicalFrame.static.addNode("TL5",TopLumbar5,positionType="Local")

        self._TopLumbar5 = TopLumbar5

    def _left_thigh_motion(self,aqui, dictRef,dictAnat,options=None):

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment("Left Thigh")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]   # reinit Technical Frame Motion ()

        # additional markers
        # NA

        # computation
                # --- LKJC
        LKJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictRef["Left Thigh"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Left Thigh"]["TF"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

            if validFrames[i]:
                LKJCvalues[i,:] = modelDecorator.VCMJointCentre( (self.mp["LeftKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=-self.mp_computed["LeftThighRotationOffset"] )

        if  "useLeftKJCmarker" in options.keys() and options["useLeftKJCmarker"] is not "LKJC":
            LOGGER.logger.info("[pyCGM2] - LKJC marker forced to use %s"%(options["useLeftKJCmarker"]))
            LKJCvalues = aqui.GetPoint(options["useLeftKJCmarker"]).GetValues()
            desc = aqui.GetPoint(options["useLeftKJCmarker"]).GetDescription()
            btkTools.smartAppendPoint(aqui,"LKJC",LKJCvalues,desc=str(desc))
        else:
            desc = seg.getReferential('TF').static.getNode_byLabel("LKJC").m_desc
            btkTools.smartAppendPoint(aqui,"LKJC",LKJCvalues,desc=str("Chord-"+desc))

        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA
        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Thigh"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictAnat["Left Thigh"]['sequence'])


            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))

    def _right_thigh_motion(self,aqui, dictRef,dictAnat,options=None):

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment("Right Thigh")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers


        RKJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))

        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right Thigh"]["TF"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

            if validFrames[i]:
                RKJCvalues[i,:] = modelDecorator.VCMJointCentre( (self.mp["RightKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["RightThighRotationOffset"] )


        if  "useRightKJCmarker" in options.keys() and options["useRightKJCmarker"] is not "RKJC":
            LOGGER.logger.info("[pyCGM2] - RKJC marker forced to use %s"%(options["useRightKJCmarker"]))
            RKJCvalues = aqui.GetPoint(options["useRightKJCmarker"]).GetValues()
            desc = aqui.GetPoint(options["useRightKJCmarker"]).GetDescription()
            btkTools.smartAppendPoint(aqui,"RKJC",RKJCvalues,desc=desc)
        else:
            desc = seg.getReferential('TF').static.getNode_byLabel("RKJC").m_desc
            #RKJCvalues = aqui.GetPoint("RKJC_Chord").GetValues()
            btkTools.smartAppendPoint(aqui,"RKJC",RKJCvalues,desc=str("Chord-"+desc))

        #btkTools.smartAppendPoint(aqui,"RKJC_Chord",RKJCvalues,desc="chord")

        # --- RKJC


        # --- motion of the anatomical referential

        # additional markers
        # NA

        # computation
        seg.anatomicalFrame.motion=[]

        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Thigh"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictAnat["Right Thigh"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))


    def _left_shank_motion(self,aqui, dictRef,dictAnat,options=None):

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment("Left Shank")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA


        # --- LAJC
        # computation
        LAJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))


        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][0])).GetValues()[i,:] #ANK
            pt2=aqui.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][1])).GetValues()[i,:] #KJC
            pt3=aqui.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][2])).GetValues()[i,:] #TIB
            ptOrigin=aqui.GetPoint(str(dictRef["Left Shank"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Left Shank"]["TF"]['sequence'])


            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

            if validFrames[i]:
                LAJCvalues[i,:] = modelDecorator.VCMJointCentre( (self.mp["LeftAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=-self.mp_computed["LeftShankRotationOffset"] )

                # update of the AJC location with rotation around abdAddAxis
                LAJCvalues[i,:] = self._rotateAjc(LAJCvalues[i,:],pt2,pt1,self.mp_computed["LeftAnkleAbAddOffset"])


        if  "useLeftAJCmarker" in options.keys() and options["useLeftAJCmarker"] is not "LAJC":
            LOGGER.logger.info("[pyCGM2] - LAJC marker forced to use %s"%(options["useLeftAJCmarker"]))
            LAJCvalues = aqui.GetPoint(options["useLeftAJCmarker"]).GetValues()
            desc = aqui.GetPoint(options["useLeftAJCmarker"]).GetDescription()
            btkTools.smartAppendPoint(aqui,"LAJC",LAJCvalues,desc=desc)
        else:
            # --- LAJC
            desc_node = seg.getReferential('TF').static.getNode_byLabel("LAJC").m_desc
            if self.mp_computed["LeftAnkleAbAddOffset"] > 0.01:
                desc="chord+AbAdRot-"+desc_node
            else:
                desc="chord "+desc_node
            btkTools.smartAppendPoint(aqui,"LAJC",LAJCvalues,desc=desc)


        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA

        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][3])).GetValues()[i,:]

            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictAnat["Left Shank"]['sequence'])
            csFrame=frame.Frame()

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))

    def _left_shankProximal_motion(self,aqui,dictAnat,options=None):

        seg=self.getSegment("Left Shank")
        segProx=self.getSegment("Left Shank Proximal")


        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        segProx.setExistFrames(validFrames)


        # --- managment of tibial torsion

        if self.m_useLeftTibialTorsion:
            tibialTorsion = -1.0*np.deg2rad(self.mp_computed["LeftTibialTorsionOffset"])
        else:
            tibialTorsion = 0.0

        if "pigStatic" in options.keys() and options["pigStatic"]:
            tibialTorsion = 0.0


        # --- motion of both technical and anatomical referentials of the proximal shank
        segProx.getReferential("TF").motion =[]
        segProx.anatomicalFrame.motion=[]


        # additional markers
        # NA

        # computation
        rotZ_tibRot = np.eye(3,3)
        rotZ_tibRot[0,0] = np.cos(tibialTorsion)
        rotZ_tibRot[0,1] = np.sin(tibialTorsion)
        rotZ_tibRot[1,0] = - np.sin(tibialTorsion)
        rotZ_tibRot[1,1] = np.cos(tibialTorsion)
        LKJC = aqui.GetPoint(str(dictAnat["Left Shank"]['labels'][3]))

        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=LKJC.GetValues()[i,:]
            segProx.getReferential("TF").addMotionFrame(seg.getReferential("TF").motion[i]) # copy technical shank

            R = np.dot(seg.anatomicalFrame.motion[i].getRotation(),rotZ_tibRot) # affect Tibial torsion to anatomical shank

            csFrame.update(R,ptOrigin)
            segProx.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))



    def _right_shank_motion(self,aqui, dictRef,dictAnat,options=None):


        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment("Right Shank")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA


        RAJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))

        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][0])).GetValues()[i,:] #ank
            pt2=aqui.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][1])).GetValues()[i,:] #kjc
            pt3=aqui.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][2])).GetValues()[i,:] #tib
            ptOrigin=aqui.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right Shank"]["TF"]['sequence'])


            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

            # ajc position from chord modified by shank offset
            if validFrames[i]:
                RAJCvalues[i,:] = modelDecorator.VCMJointCentre( (self.mp["RightAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["RightShankRotationOffset"] )
            # update of the AJC location with rotation around abdAddAxis
                RAJCvalues[i,:] = self._rotateAjc(RAJCvalues[i,:],pt2,pt1,   self.mp_computed["RightAnkleAbAddOffset"])

        # --- LAJC

        if  "useRightAJCmarker" in options.keys() and options["useRightAJCmarker"] is not "RAJC":
            LOGGER.logger.info("[pyCGM2] - RAJC marker forced to use %s"%(options["useRightAJCmarker"]))
            RAJCvalues = aqui.GetPoint(options["useRightAJCmarker"]).GetValues()
            desc = aqui.GetPoint(options["useRightAJCmarker"]).GetDescription()
            btkTools.smartAppendPoint(aqui,"RAJC",RAJCvalues,desc=desc)
        else:
            # --- RAJC
            desc_node = seg.getReferential('TF').static.getNode_byLabel("RAJC").m_desc
            if self.mp_computed["RightAnkleAbAddOffset"] >0.01:
                desc="chord+AbAdRot-"+desc_node
            else:
                desc="chord"+desc_node

            btkTools.smartAppendPoint(aqui,"RAJC",RAJCvalues,desc=desc)

        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]

        # additional markers
        # NA

        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictAnat["Right Shank"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))

    def _right_shankProximal_motion(self,aqui,dictAnat,options=None):

        seg=self.getSegment("Right Shank")
        segProx=self.getSegment("Right Shank Proximal")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        segProx.setExistFrames(validFrames)

        # --- management of the tibial torsion
        if self.m_useRightTibialTorsion:
            tibialTorsion = np.deg2rad(self.mp_computed["RightTibialTorsionOffset"])
        else:
            tibialTorsion = 0.0


        # --- motion of both technical and anatomical referentials of the proximal shank
        segProx.getReferential("TF").motion =[]
        segProx.anatomicalFrame.motion=[]

        # additional markers
        # NA

        # computation
        rotZ_tibRot = np.eye(3,3)
        rotZ_tibRot[0,0] = np.cos(tibialTorsion)
        rotZ_tibRot[0,1] = np.sin(tibialTorsion)
        rotZ_tibRot[1,0] = - np.sin(tibialTorsion)
        rotZ_tibRot[1,1] = np.cos(tibialTorsion)

        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Shank"]['labels'][3])).GetValues()[i,:]

            segProx.getReferential("TF").addMotionFrame(seg.getReferential("TF").motion[i]) # copy technical shank

            R = np.dot(seg.anatomicalFrame.motion[i].getRotation(),rotZ_tibRot)

            csFrame.update(R,ptOrigin)
            segProx.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))





    def _left_foot_motion(self,aqui, dictRef,dictAnat,options=None):

        seg=self.getSegment("Left Foot")
        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][0])).GetValues()[i,:] #toe
            pt2=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][1])).GetValues()[i,:] #ajc

            if dictRef["Left Foot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][2])).GetValues()[i,:]
                v=(pt3-pt1)
            else:
                v=self.getSegment("Left Shank Proximal").anatomicalFrame.motion[i].m_axisY

            ptOrigin=aqui.GetPoint(str(dictRef["Left Foot"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Left Foot"]["TF"]['sequence'])


            if "viconCGM1compatible" in options.keys() and options["viconCGM1compatible"]:
                R2 = R
            else:
                R2 = np.dot(R,self._R_leftUnCorrfoot_dist_prox)


            csFrame.m_axisX=R2[:,0]
            csFrame.m_axisY=R2[:,1]
            csFrame.m_axisZ=R2[:,2]
            csFrame.setRotation(R2)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))


        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA

        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Left Foot"]['labels'][3])).GetValues()[i,:]

            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)

            csFrame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))



    def _right_foot_motion(self,aqui, dictRef,dictAnat,options=None):

        seg=self.getSegment("Right Foot")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][0])).GetValues()[i,:] #toe
            pt2=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][1])).GetValues()[i,:] #ajc

            if dictRef["Right Foot"]["TF"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][2])).GetValues()[i,:]
                v=(pt3-pt1)
            else:
                v=self.getSegment("Right Shank Proximal").anatomicalFrame.motion[i].m_axisY

            ptOrigin=aqui.GetPoint(str(dictRef["Right Foot"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right Foot"]["TF"]['sequence'])

            if "viconCGM1compatible" in options.keys() and options["viconCGM1compatible"]:
                R2 = R # e.g from proximal shank
            else:
                R2 = np.dot(R,self._R_rightUnCorrfoot_dist_prox) # e.g from distal shank Y axis

            csFrame.m_axisX=R2[:,0]
            csFrame.m_axisY=R2[:,1]
            csFrame.m_axisZ=R2[:,2]
            csFrame.setRotation(R2)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))


        # --- motion of the anatomical referential

        # additional markers
        # NA

        # computation
        seg.anatomicalFrame.motion=[]

        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Right Foot"]['labels'][3])).GetValues()[i,:]

            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)

            csFrame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))

    # ---- static PIG -----

    def _left_foot_motion_static(self,aquiStatic, dictAnat,options=None):


        seg=self.getSegment("Left Foot")

        validFrames = btkTools.getValidFrames(aquiStatic,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA


        # computation
        csFrame=frame.Frame()
        for i in range(0,aquiStatic.GetPointFrameNumber()):
            ptOrigin=aquiStatic.GetPoint(str(dictAnat["Left Foot"]['labels'][3])).GetValues()[i,:]


            pt1=aquiStatic.GetPoint(str(dictAnat["Left Foot"]['labels'][0])).GetValues()[i,:] #toe
            pt2=aquiStatic.GetPoint(str(dictAnat["Left Foot"]['labels'][1])).GetValues()[i,:] #hee

            if ("leftFlatFoot" in options.keys() and options["leftFlatFoot"]):
                pt2[2] = pt1[2]+self.mp['LeftSoleDelta']

            if dictAnat["Left Foot"]['labels'][2] is not None:
                pt3=aquiStatic.GetPoint(str(dictAnat["Left Foot"]['labels'][2])).GetValues()[i,:]
                v=(pt3-pt1)
            else:
                v=self.getSegment("Left Shank").anatomicalFrame.motion[i].m_axisY # distal segment

            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))


            x,y,z,R=frame.setFrameData(a1,a2,dictAnat["Left Foot"]['sequence'])

            csFrame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))


    def _right_foot_motion_static(self,aquiStatic, dictAnat,options=None):

        seg=self.getSegment("Right Foot")


        validFrames = btkTools.getValidFrames(aquiStatic,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA

        # computation
        csFrame=frame.Frame()
        for i in range(0,aquiStatic.GetPointFrameNumber()):
            ptOrigin=aquiStatic.GetPoint(str(dictAnat["Right Foot"]['labels'][3])).GetValues()[i,:]


            pt1=aquiStatic.GetPoint(str(dictAnat["Right Foot"]['labels'][0])).GetValues()[i,:] #toe
            pt2=aquiStatic.GetPoint(str(dictAnat["Right Foot"]['labels'][1])).GetValues()[i,:] #hee

            if ("rightFlatFoot" in options.keys() and options["rightFlatFoot"]):
                pt2[2] = pt1[2]+self.mp['RightSoleDelta']


            if dictAnat["Right Foot"]['labels'][2] is not None:
                pt3=aquiStatic.GetPoint(str(dictAnat["Right Foot"]['labels'][2])).GetValues()[i,:]
                v=(pt3-pt1)
            else:
                v=self.getSegment("Right Shank").anatomicalFrame.motion[i].m_axisY # distal segment

            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictAnat["Right Foot"]['sequence'])


            csFrame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))

    # ----- least-square Segmental motion ------
    def _pelvis_motion_optimize(self,aqui, dictRef, motionMethod,anatomicalFrameMotionEnable=True):

        seg=self.getSegment("Pelvis")
        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        #  --- check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            btkTools.isPointsExist(aqui,seg.m_tracking_markers)

        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

        # part 1: get global location in Static
        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.zeros((len(seg.m_tracking_markers),3))
            i=0
            for label in seg.m_tracking_markers:
                staticPos[i,:] = seg.getReferential("TF").static.getNode_byLabel(label).m_global
                i+=1

        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1

            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt

                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)
                csFrame.m_axisX=R[:,0]
                csFrame.m_axisY=R[:,1]
                csFrame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))


        # --- HJC
        values_LHJCnode=seg.getReferential('TF').getNodeTrajectory("LHJC")
        values_RHJCnode=seg.getReferential('TF').getNodeTrajectory("RHJC")

        for i in range(0,aqui.GetPointFrameNumber()):
            if not validFrames[i]:
                values_LHJCnode[i,:] = np.zeros(3)
                values_RHJCnode[i,:] = np.zeros(3)


        desc_L = seg.getReferential('TF').static.getNode_byLabel("LHJC").m_desc
        desc_R = seg.getReferential('TF').static.getNode_byLabel("RHJC").m_desc
        btkTools.smartAppendPoint(aqui,"LHJC",values_LHJCnode, desc=str("opt-"+desc_L))
        btkTools.smartAppendPoint(aqui,"RHJC",values_RHJCnode, desc=str("opt-"+desc_R))

        # --- midASIS
        values_midASISnode = seg.getReferential('TF').getNodeTrajectory("midASIS")
        btkTools.smartAppendPoint(aqui,"midASIS",values_midASISnode, desc="opt")

        # midHJC
        val=(aqui.GetPoint("LHJC").GetValues() + aqui.GetPoint("RHJC").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,"midHJC",val,desc="opt")


    def _left_thigh_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Left Thigh")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        #  --- add LHJC if list <2 - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "LHJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("LHJC")
                    LOGGER.logger.debug("LHJC added to tracking marker list")

            btkTools.isPointsExist(aqui,seg.m_tracking_markers)


        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

        # part 1: get back static global position ( look out i use nodes)

        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.zeros((len(seg.m_tracking_markers),3))
            i=0
            for label in seg.m_tracking_markers: # recupere les tracki
                staticPos[i,:] = seg.getReferential("TF").static.getNode_byLabel(label).m_global
                i+=1


        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1


            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt


                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)
                csFrame.m_axisX=R[:,0]
                csFrame.m_axisY=R[:,1]
                csFrame.m_axisZ=R[:,2]

                seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

        # --- LKJC
        desc = seg.getReferential('TF').static.getNode_byLabel("LKJC").m_desc
        values_LKJCnode=seg.getReferential('TF').getNodeTrajectory("LKJC")

        for i in range(0,aqui.GetPointFrameNumber()):
            if not validFrames[i]:
                values_LKJCnode[i,:] = np.zeros(3)

        btkTools.smartAppendPoint(aqui,"LKJC",values_LKJCnode, desc=str("opt-"+desc))


        # --- LHJC from Thigh
        # values_HJCnode=seg.getReferential('TF').getNodeTrajectory("LHJC")
        # btkTools.smartAppendPoint(aqui,"LHJC-Thigh",values_HJCnode, desc="opt from Thigh")

        # remove LHC from list of tracking markers
        if "LHJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("LHJC")




    def _right_thigh_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Right Thigh")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        #  --- add RHJC if list <2 - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "RHJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("RHJC")
                    LOGGER.logger.debug("RHJC added to tracking marker list")

        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

        # part 1: get back static global position ( look ou i use nodes)

        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.zeros((len(seg.m_tracking_markers),3))
            i=0
            for label in seg.m_tracking_markers:
                staticPos[i,:] = seg.getReferential("TF").static.getNode_byLabel(label).m_global
                i+=1

        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1


            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt


                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)
                csFrame.m_axisX=R[:,0]
                csFrame.m_axisY=R[:,1]
                csFrame.m_axisZ=R[:,2]

                seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

        # --- RKJC
        desc = seg.getReferential('TF').static.getNode_byLabel("RKJC").m_desc
        values_RKJCnode=seg.getReferential('TF').getNodeTrajectory("RKJC")

        for i in range(0,aqui.GetPointFrameNumber()):
            if not validFrames[i]:
                values_RKJCnode[i,:] = np.zeros(3)

        btkTools.smartAppendPoint(aqui,"RKJC",values_RKJCnode, desc=str("opt-"+desc))

        # --- RHJC from Thigh
        #values_HJCnode=seg.getReferential('TF').getNodeTrajectory("RHJC")
        #btkTools.smartAppendPoint(aqui,"RHJC-Thigh",values_HJCnode, desc="opt from Thigh")

        # remove HJC from list of tracking markers
        if "RHJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("RHJC")


    def _left_shank_motion_optimize(self,aqui, dictRef,  motionMethod):

        seg=self.getSegment("Left Shank")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        #  --- add LKJC if list <2 - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "LKJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("LKJC")
                    LOGGER.logger.debug("LKJC added to tracking marker list")

        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

       # part 1: get back static global position ( look ou i use nodes)
        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.zeros((len(seg.m_tracking_markers),3))
            i=0
            for label in seg.m_tracking_markers:
                staticPos[i,:] = seg.getReferential("TF").static.getNode_byLabel(label).m_global
                i+=1

        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1


            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt

                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)
                csFrame.m_axisX=R[:,0]
                csFrame.m_axisY=R[:,1]
                csFrame.m_axisZ=R[:,2]

                seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))


        # --- LAJC
        desc = seg.getReferential('TF').static.getNode_byLabel("LAJC").m_desc
        values_LAJCnode=seg.getReferential('TF').getNodeTrajectory("LAJC")

        for i in range(0,aqui.GetPointFrameNumber()):
            if not validFrames[i]:
                values_LAJCnode[i,:] = np.zeros(3)

        btkTools.smartAppendPoint(aqui,"LAJC",values_LAJCnode, desc=str("opt"+desc))

        # --- KJC from Shank
        #values_KJCnode=seg.getReferential('TF').getNodeTrajectory("LKJC")
        #btkTools.smartAppendPoint(aqui,"LKJC-Shank",values_KJCnode, desc="opt from Shank")


        # remove KJC from list of tracking markers
        if "LKJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("LKJC")



    def _right_shank_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Right Shank")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        #  --- add RKJC if list <2 - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "RKJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("RKJC")
                    LOGGER.logger.debug("RKJC added to tracking marker list")

        # --- Motion of the Technical frame

        seg.getReferential("TF").motion =[]

        # part 1: get back static global position ( look ou i use nodes)

        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.zeros((len(seg.m_tracking_markers),3))
            i=0
            for label in seg.m_tracking_markers:
                staticPos[i,:] = seg.getReferential("TF").static.getNode_byLabel(label).m_global
                i+=1

        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1


            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt


                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)
                csFrame.m_axisX=R[:,0]
                csFrame.m_axisY=R[:,1]
                csFrame.m_axisZ=R[:,2]

                seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

        # RAJC
        desc = seg.getReferential('TF').static.getNode_byLabel("RAJC").m_desc
        values_RAJCnode=seg.getReferential('TF').getNodeTrajectory("RAJC")

        for i in range(0,aqui.GetPointFrameNumber()):
            if not validFrames[i]:
                values_RAJCnode[i,:] = np.zeros(3)

        btkTools.smartAppendPoint(aqui,"RAJC",values_RAJCnode, desc=str("opt-"+desc))



        # --- KJC from Shank
        #values_KJCnode=seg.getReferential('TF').getNodeTrajectory("RKJC")
        #btkTools.smartAppendPoint(aqui,"RKJC-Shank",values_KJCnode, desc="opt from Shank")

        # remove KJC from list of tracking markers
        if "RKJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("RKJC")

    def _leftFoot_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Left Foot")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        #  --- add LAJC if marker list < 2  - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "LAJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("LAJC")
                    LOGGER.logger.debug("LAJC added to tracking marker list")

        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

        # part 1: get global location in Static
        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.zeros((len(seg.m_tracking_markers),3))
            i=0
            for label in seg.m_tracking_markers:
                staticPos[i,:] = seg.getReferential("TF").static.getNode_byLabel(label).m_global
                i+=1

        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1

            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm= motion.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt


                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)
                csFrame.m_axisX=R[:,0]
                csFrame.m_axisY=R[:,1]
                csFrame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))


        # --- AJC from Foot
        values_AJCnode=seg.getReferential('TF').getNodeTrajectory("LAJC")
        for i in range(0,aqui.GetPointFrameNumber()):
            if not validFrames[i]:
                values_AJCnode[i,:] = np.zeros(3)
        #btkTools.smartAppendPoint(aqui,"LAJC-Foot",values_AJCnode, desc="opt from Foot")


        # remove AJC from list of tracking markers
        if "LAJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("LAJC")


    def _rightFoot_motion_optimize(self,aqui, dictRef, motionMethod):

        seg=self.getSegment("Right Foot")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        #  --- add RAJC if marker list < 2  - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if "RAJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append("RAJC")
                    LOGGER.logger.debug("RAJC added to tracking marker list")

        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

        # part 1: get global location in Static
        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.zeros((len(seg.m_tracking_markers),3))
            i=0
            for label in seg.m_tracking_markers:
                staticPos[i,:] = seg.getReferential("TF").static.getNode_byLabel(label).m_global
                i+=1

        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            if seg.m_tracking_markers != []: # work with traking markers
                dynPos = np.zeros((len(seg.m_tracking_markers),3)) # use
                k=0
                for label in seg.m_tracking_markers:
                    dynPos[k,:] = aqui.GetPoint(label).GetValues()[i,:]
                    k+=1

            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm= motion.segmentalLeastSquare(staticPos,
                                                              dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt


                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)
                csFrame.m_axisX=R[:,0]
                csFrame.m_axisY=R[:,1]
                csFrame.m_axisZ=R[:,2]

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))


        # --- AJC from Foot
        values_AJCnode=seg.getReferential('TF').getNodeTrajectory("RAJC")
        for i in range(0,aqui.GetPointFrameNumber()):
            if not validFrames[i]:
                values_AJCnode[i,:] = np.zeros(3)
        #btkTools.smartAppendPoint(aqui,"RAJC-Foot",values_AJCnode, desc="opt from Foot")


        # remove AJC from list of tracking markers
        if "RAJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove("RAJC")


    def _anatomical_motion(self,aqui,segmentLabel,originLabel=""):

        seg=self.getSegment(segmentLabel)

        # --- Motion of the Anatomical frame
        seg.anatomicalFrame.motion=[]


        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(originLabel).GetValues()[i,:]
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            csFrame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))


    def _rotate_anatomical_motion(self,segmentLabel,angle,aqui,options=None):

        seg=self.getSegment(segmentLabel)


        angle = np.deg2rad(angle)

        # --- motion of both technical and anatomical referentials of the proximal shank

        #seg.anatomicalFrame.motion=[]

        # additional markers
        # NA

        # computation
        rotZ = np.eye(3,3)
        rotZ[0,0] = np.cos(angle)
        rotZ[0,1] = -np.sin(angle)
        rotZ[1,0] =  np.sin(angle)
        rotZ[1,1] = np.cos(angle)

        for i in range(0,aqui.GetPointFrameNumber()):

            ptOrigin=seg.anatomicalFrame.motion[i].getTranslation()

            R = np.dot(seg.anatomicalFrame.motion[i].getRotation(),rotZ)

            seg.anatomicalFrame.motion[i].update(R,ptOrigin)




    # ---- tools ----
    def _rotateAjc(self,ajc,kjc,ank, offset):



        a1=(kjc-ajc)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(ank-ajc)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,"ZXY")
        csFrame=frame.Frame()

        csFrame.m_axisX=x
        csFrame.m_axisY=y
        csFrame.m_axisZ=z
        csFrame.setRotation(R)
        csFrame.setTranslation(ank)

        loc=np.dot(R.T,ajc-ank)

        abAdangle = np.deg2rad(offset)

        rotAbdAdd = np.array([[1, 0, 0],[0, np.cos(abAdangle), -1.0*np.sin(abAdangle)], [0, np.sin(abAdangle), np.cos(abAdangle) ]])

        finalRot= np.dot(R,rotAbdAdd)

        return  np.dot(finalRot,loc)+ank


# ---- Technical Referential Calibration
    def _head_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):

        pfn = aquiStatic.GetPointFrameNumber()


        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment("Head")
        seg.resetMarkerLabels()

        # ---  additional markers and Update of the marker segment list
        # new markers (head)
        valmFH=(aquiStatic.GetPoint("LFHD").GetValues() + aquiStatic.GetPoint("RFHD").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"midFH",valmFH,desc="")

        valmBH=(aquiStatic.GetPoint("LBHD").GetValues() + aquiStatic.GetPoint("RBHD").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"midBH",valmBH,desc="")


        valmLH=(aquiStatic.GetPoint("LFHD").GetValues() + aquiStatic.GetPoint("LBHD").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"midLH",valmLH,desc="")


        valmHC=(valmFH+valmBH) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"HC",valmHC,desc="")

        seg.addCalibrationMarkerLabel("midFH")
        seg.addCalibrationMarkerLabel("midBH")
        seg.addCalibrationMarkerLabel("HC")
        seg.addCalibrationMarkerLabel("midLH")

        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictRef["Head"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Head"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef["Head"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Head"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Head"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # head straight
        y2=y
        x2 = np.cross(y,np.array([0,0,1]))
        z2 = np.cross(x2,y)
        straightHead=np.array([x2,y2,z2]).T
        relativeR = np.dot(R.T,straightHead)
        angle_y,angle_x,angle_z = euler.euler_yxz(relativeR)


        if ("headFlat" in options.keys() and options["headFlat"]):
            LOGGER.logger.debug ("option (headFlat) enable")
            self.mp_computed["HeadOffset"] =  np.rad2deg(angle_y)
        else:
            self.mp_computed["HeadOffset"] =  0



        #nodes
        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        for label in seg.m_calibration_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        lfhd = tf.static.getNode_byLabel("LFHD").m_local
        rfhd = tf.static.getNode_byLabel("RFHD").m_local

        seg.m_info["headScaleAdjustment"] =2
        seg.m_info["headScale"] =(np.linalg.norm(lfhd-rfhd) - markerDiameter) * seg.m_info["headScaleAdjustment"]

    def _head_AnatomicalCalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd, options=None):

        seg=self.getSegment("Head")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Head"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Head"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Head"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Head"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Head"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # # --- relative rotation Technical Anatomical
        offset =  1.0*np.deg2rad(self.mp_computed["HeadOffset"])

        rot = np.eye(3,3)
        rot[0,0] = np.cos(offset)
        rot[0,2] =  np.sin(offset)
        rot[2,0] = - np.sin(offset)
        rot[2,2] = np.cos(offset)
        #

        R2 = np.dot(R,rot)
        seg.anatomicalFrame.static.setRotation(R2)

        tf = seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))


        # --- node manager
        for node in seg.getReferential("TF").static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

        headCoM = np.array([ seg.m_info["headScale"]*0.52,0.0,0.0])
        SkullOriginOffset = np.array([ -0.84, 0, -0.3 ])
        headCoM = headCoM + SkullOriginOffset * seg.m_info["headScale"]

        # length - com
        c7 = seg.anatomicalFrame.static.getNode_byLabel("C7").m_local
        seg.setLength(np.linalg.norm(c7-headCoM)/2.0)
        seg.anatomicalFrame.static.addNode("com",headCoM,positionType="Local")

        seg.anatomicalFrame.static.addNode("SkullOriginOffset", SkullOriginOffset * seg.m_info["headScale"],positionType="Local")



    def _torso_calibrate(self,aquiStatic, dictRef,frameInit,frameEnd, options=None):

        pfn = aquiStatic.GetPointFrameNumber()



        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment("Thorax")
        seg.resetMarkerLabels()

        # ---  additional markers and Update of the marker segment list
        valTop=(aquiStatic.GetPoint("CLAV").GetValues() + aquiStatic.GetPoint("C7").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"midTop",valTop,desc="")

        valBottom=(aquiStatic.GetPoint("STRN").GetValues() + aquiStatic.GetPoint("T10").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"midBottom",valBottom,desc="")

        valFront=(aquiStatic.GetPoint("STRN").GetValues() + aquiStatic.GetPoint("CLAV").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"midFront",valFront,desc="")


        seg.addCalibrationMarkerLabel("midTop")
        seg.addCalibrationMarkerLabel("midBottom")
        seg.addCalibrationMarkerLabel("midFront")

        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Thorax"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        OT = ptOrigin + -1.0*(markerDiameter/2.0)* tf.static.m_axisX
        clav = aquiStatic.GetPoint("CLAV").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        offset =( clav - OT)*1.05
        c7offset = aquiStatic.GetPoint(str("C7")).GetValues()[frameInit:frameEnd,:].mean(axis=0) + offset

        btkTools.smartAppendPoint(aquiStatic,"OT", OT* np.ones((pfn,3)), desc="")
        btkTools.smartAppendPoint(aquiStatic,"C7o", c7offset* np.ones((pfn,3)), desc="")


        seg.addCalibrationMarkerLabel("OT")
        seg.addCalibrationMarkerLabel("C7o")
        #tf.static.addNode("C7o",c7,positionType="Global",desc ="")


        if self.m_bodypart is not enums.BodyPart.LowerLimbTrunk:
            # shoulder joints
            LSHO=aquiStatic.GetPoint(str("LSHO")).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            LVWM = np.cross((LSHO - OT ),tf.static.m_axisX ) + LSHO

            btkTools.smartAppendPoint(aquiStatic,"LVWM", LVWM* np.ones((pfn,3)),desc="")
            LSJC = modelDecorator.VCMJointCentre( -1.0* (self.mp["LeftShoulderOffset"]+ markerDiameter/2.0),LSHO,OT,LVWM, beta=0 )

            RSHO=aquiStatic.GetPoint(str("RSHO")).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            RVWM = np.cross(( tf.static.m_axisX ),( OT-RSHO )) + RSHO
            btkTools.smartAppendPoint(aquiStatic,"RVWM", RVWM* np.ones((pfn,3)),desc="")
            RSJC =  modelDecorator.VCMJointCentre( self.mp["RightShoulderOffset"]+ markerDiameter/2.0 ,RSHO,OT,RVWM, beta=0 )

            # left
            if tf.static.isNodeExist("LSJC"):
                nodeLSJC = tf.static.getNode_byLabel("LSJC")
            else:
                tf.static.addNode("LSJC_cgm1",LSJC,positionType="Global",desc = "chord")
                tf.static.addNode("LSJC",LSJC,positionType="Global",desc = "chord")
                nodeLSJC = tf.static.getNode_byLabel("LSJC")

            btkTools.smartAppendPoint(aquiStatic,"LSJC",
                        nodeLSJC.m_global* np.ones((pfn,3)),
                        desc=nodeLSJC.m_desc)

            # right
            if tf.static.isNodeExist("RSJC"):
                nodeRSJC = tf.static.getNode_byLabel("RSJC")
            else:
                tf.static.addNode("RSJC_cgm1",RSJC,positionType="Global",desc = "chord")
                tf.static.addNode("RSJC",RSJC,positionType="Global",desc = "chord")
                nodeRSJC = tf.static.getNode_byLabel("RSJC")

            btkTools.smartAppendPoint(aquiStatic,"RSJC",
                        nodeRSJC.m_global* np.ones((pfn,3)),
                        desc=nodeRSJC.m_desc)


        #nodes
        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        for label in seg.m_calibration_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # TL5
        if self.m_bodypart !=  enums.BodyPart.UpperLimb:
            TL5 = self.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("TL5").m_global
            tf.static.addNode("TL5",TL5,positionType="Global")






    def _torso_Anatomicalcalibrate(self,aquiStatic, dictAnatomic,frameInit,frameEnd):

        seg=self.getSegment("Thorax")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic["Thorax"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic["Thorax"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic["Thorax"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Thorax"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #OT

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Thorax"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))


        # --- node manager
        for node in seg.getReferential("TF").static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

        # length
        if self.m_bodypart !=  enums.BodyPart.UpperLimb:
            lsjc = seg.anatomicalFrame.static.getNode_byLabel("LSJC").m_global
            rsjc = seg.anatomicalFrame.static.getNode_byLabel("RSJC").m_global
            c7o = seg.anatomicalFrame.static.getNode_byLabel("C7o").m_global

            l5 = self.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("TL5").m_global
            length = np.linalg.norm(l5-c7o)
            seg.setLength(length)

            seg.m_info["Scale"] = np.mean((np.linalg.norm(ptOrigin-rsjc),
                                            np.linalg.norm(ptOrigin-lsjc)))


            # com not computed there but durinf motion !!
            com = (c7o + ( l5 - c7o ) * 0.63 )
            seg.anatomicalFrame.static.addNode("comStatic",com,positionType="Global")

        else:
            top = seg.anatomicalFrame.static.getNode_byLabel("midTop").m_local
            bottom = seg.anatomicalFrame.static.getNode_byLabel("midBottom").m_local
            seg.setLength(np.linalg.norm(top-bottom))
            seg.m_info["Scale"] = np.linalg.norm(top-bottom)


    def _clavicle_calibrate(self,side, aquiStatic, dictRef,frameInit,frameEnd, options=None):

        if side == "Left":
            prefix ="L"
        if side == "Right":
            prefix ="R"
            s= 1.0

        segname = side + " " +"Clavicle"
        seg=self.getSegment(segname)
        seg.resetMarkerLabels()

        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictRef[segname]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef[segname]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef[segname]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictRef[segname]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef[segname]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)


    def _clavicle_Anatomicalcalibrate(self,side,aquiStatic, dictAnatomic,frameInit,frameEnd):

        if side == "Left":
            prefix ="L"
        if side == "Right":
            prefix ="R"
            s= 1.0

        seg=self.getSegment(side+" Clavicle")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic[side+" Clavicle"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#
        pt2=aquiStatic.GetPoint(str(dictAnatomic[side+" Clavicle"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#
        pt3=aquiStatic.GetPoint(str(dictAnatomic[side+" Clavicle"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic[side+" Clavicle"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic[side+" Clavicle"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))


        # --- node manager
        # for node in seg.getReferential("TF").static.getNodes():
        #     seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())
        #


    def _constructArmVirtualMarkers(self,side, aqui):

        if side == "Left":
            prefix ="L"
            s= -1.0
        if side == "Right":
            prefix ="R"
            s= 1.0

        # mid wrist
        midwrist=(aqui.GetPoint(prefix+"WRA").GetValues() + aqui.GetPoint(prefix+"WRB").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,prefix+"MWP",midwrist,desc="")

        # virtual wand ( need SJC !!)
        pfn = aqui.GetPointFrameNumber()
        SJC = aqui.GetPoint(prefix+"SJC").GetValues()

        LHE=aqui.GetPoint(prefix+"ELB").GetValues()
        MWP=aqui.GetPoint(prefix+"MWP").GetValues()

        CVMvalues = np.zeros((pfn,3))

        for i in range(0,pfn):
            CVM = s*np.cross((MWP[i,:]-LHE[i,:]),(SJC[i,:]-LHE[i,:]))
            CVM = CVM / np.linalg.norm(CVM)
            CVMvalues[i,:] = LHE[i,:] + 50.0*CVM

        btkTools.smartAppendPoint(aqui,prefix+"CVM", CVMvalues, desc="")


    def _upperArm_calibrate(self,side, aquiStatic, dictRef,frameInit,frameEnd, options=None):

        if side == "Left":
            prefix ="L"
        if side == "Right":
            prefix ="R"
            s= 1.0

        pfn = aquiStatic.GetPointFrameNumber()

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        segname = side + " " +"UpperArm"
        seg=self.getSegment(segname)
        seg.resetMarkerLabels()

        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictRef[segname]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #ELB
        pt2=aquiStatic.GetPoint(str(dictRef[segname]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #SJC
        pt3=aquiStatic.GetPoint(str(dictRef[segname]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #CVM

        ptOrigin=aquiStatic.GetPoint(str(dictRef[segname]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #ELB

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef[segname]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # comuptation of EJC ( need virtual wand)
        EJC =  modelDecorator.VCMJointCentre( (self.mp[side+"ElbowWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=0 ) #ELB,SJC,CVM

        if tf.static.isNodeExist(prefix +"EJC"):
            nodeEJC = tf.static.getNode_byLabel(prefix+"EJC")
        else:
            tf.static.addNode(prefix+"EJC_cgm1",EJC,positionType="Global",desc = "chord")
            tf.static.addNode(prefix+"EJC",EJC,positionType="Global",desc = "chord")
            nodeEJC = tf.static.getNode_byLabel(prefix+"EJC")

        btkTools.smartAppendPoint(aquiStatic,prefix+"EJC",
                    nodeEJC.m_global* np.ones((pfn,3)),
                    desc=nodeEJC.m_desc)

        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        for label in seg.m_calibration_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")


    def _upperArm_Anatomicalcalibrate(self,side, aquiStatic, dictAnatomic,frameInit,frameEnd):

        if side == "Left":
            prefix ="L"
        if side == "Right":
            prefix ="R"
            s= 1.0

        segname = side + " " +"UpperArm"

        seg=self.getSegment(segname)

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic[segname]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #EJC
        pt2=aquiStatic.GetPoint(str(dictAnatomic[segname]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #SJC
        pt3=aquiStatic.GetPoint(str(dictAnatomic[segname]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #WJC

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic[segname]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic[segname]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))


        # --- node manager
        for node in seg.getReferential("TF").static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

        # length
        top = seg.anatomicalFrame.static.getNode_byLabel(prefix+"SJC").m_local
        bottom = seg.anatomicalFrame.static.getNode_byLabel(prefix+"EJC").m_local
        seg.setLength(np.linalg.norm(top-bottom))

    def _foreArm_calibrate(self,side,aquiStatic, dictRef,frameInit,frameEnd, options=None):

        if side == "Left":
            prefix ="L"
            s= -1.0
        if side == "Right":
            prefix ="R"
            s= 1.0

        pfn = aquiStatic.GetPointFrameNumber()

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment(side +" ForeArm")
        seg.resetMarkerLabels()

        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictRef[side +" ForeArm"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#US (WRB)
        pt2=aquiStatic.GetPoint(str(dictRef[side +" ForeArm"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#EJC
        pt3=aquiStatic.GetPoint(str(dictRef[side +" ForeArm"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#RS (WRW)

        ptOrigin=aquiStatic.GetPoint(str(dictRef[side +" ForeArm"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef[side +" ForeArm"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # comuptation of WJC ( need virtual wand)
        EJC = pt2
        US=pt3
        RS=pt1

        MWP=aquiStatic.GetPoint(prefix+"MWP").GetValues()[frameInit:frameEnd,:].mean(axis=0)

        WJCaxis = np.cross((US-RS),(EJC-MWP))
        WJCaxis = WJCaxis / np.linalg.norm(WJCaxis)
        WJC =MWP +  (s*(self.mp[side +"WristWidth"]+markerDiameter)/2.0)*WJCaxis

        if tf.static.isNodeExist(prefix+"WJC"):
            nodeWJC = tf.static.getNode_byLabel(prefix+"WJC")
        else:
            tf.static.addNode(prefix+"WJC_cgm1",WJC,positionType="Global",desc = "midCgm1")
            tf.static.addNode(prefix+"WJC",WJC,positionType="Global",desc = "midCgm1")
            nodeWJC = tf.static.getNode_byLabel(prefix+"WJC")

        btkTools.smartAppendPoint(aquiStatic,prefix+"WJC",
                    nodeWJC.m_global* np.ones((pfn,3)),
                    desc=nodeWJC.m_desc)

        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        for label in seg.m_calibration_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

    def _foreArm_Anatomicalcalibrate(self,side,aquiStatic, dictAnatomic,frameInit,frameEnd):

        if side == "Left":
            prefix ="L"
        if side == "Right":
            prefix ="R"
            s= 1.0

        seg=self.getSegment(side+" ForeArm")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic[side+" ForeArm"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#WJC
        pt2=aquiStatic.GetPoint(str(dictAnatomic[side+" ForeArm"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#EJC

        if dictAnatomic[side+" ForeArm"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic[side+" ForeArm"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#Not used

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic[side+" ForeArm"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        if dictAnatomic[side+" ForeArm"]['labels'][2] is not None:
            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))
        else:
            v=self.getSegment(side+ " UpperArm").anatomicalFrame.static.m_axisY

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic[side+" ForeArm"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))


        # --- node manager
        for node in seg.getReferential("TF").static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

        # length
        top = seg.anatomicalFrame.static.getNode_byLabel(prefix+"EJC").m_local
        bottom = seg.anatomicalFrame.static.getNode_byLabel(prefix+"WJC").m_local
        seg.setLength(np.linalg.norm(top-bottom))


    def _hand_calibrate(self,side, aquiStatic, dictRef,frameInit,frameEnd, options=None):

        if side == "Left":
            prefix ="L"
        if side == "Right":
            prefix ="R"
            s= 1.0

        pfn = aquiStatic.GetPointFrameNumber()

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment(side +" Hand")
        seg.resetMarkerLabels()

        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        #   referential construction
        pt1=aquiStatic.GetPoint(str(dictRef[side +" Hand"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#FIN
        pt2=aquiStatic.GetPoint(str(dictRef[side +" Hand"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#WJC
        pt3=aquiStatic.GetPoint(str(dictRef[side +" Hand"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#MWP

        ptOrigin=aquiStatic.GetPoint(str(dictRef[side +" Hand"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef[side +" Hand"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # comuptation of Hand Origin
        HO =  modelDecorator.VCMJointCentre( (self.mp[side+"HandThickness"]+ markerDiameter)/2.0 ,pt1, pt2, pt3, beta=0 )

        tf.static.addNode(prefix+"HO",HO,positionType="Global",desc = "ch1-handOrigin")
        nodeHO = tf.static.getNode_byLabel(prefix+"HO")

        btkTools.smartAppendPoint(aquiStatic,prefix +"HO",
                nodeHO.m_global* np.ones((pfn,3)),
                desc=nodeHO.m_desc)

        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        for label in seg.m_calibration_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")


    def _hand_Anatomicalcalibrate(self,side,aquiStatic, dictAnatomic,frameInit,frameEnd):

        if side == "Left":
            prefix ="L"
        if side == "Right":
            prefix ="R"


        seg=self.getSegment(side +" Hand")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic[side +" Hand"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # HO
        pt2=aquiStatic.GetPoint(str(dictAnatomic[side +" Hand"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #WJC
        pt3=aquiStatic.GetPoint(str(dictAnatomic[side +" Hand"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0) #MWP

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic[side +" Hand"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # HO

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic[side +" Hand"]['sequence'])

        seg.anatomicalFrame.static.m_axisX=x
        seg.anatomicalFrame.static.m_axisY=y
        seg.anatomicalFrame.static.m_axisZ=z
        seg.anatomicalFrame.static.setRotation(R)
        seg.anatomicalFrame.static.setTranslation(ptOrigin)

        # --- relative rotation Technical Anatomical
        tf=seg.getReferential("TF")
        tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

        # --- node manager
        for node in seg.getReferential("TF").static.getNodes():
            seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

        # length
        top = seg.anatomicalFrame.static.getNode_byLabel(prefix+"WJC").m_local
        bottom = seg.anatomicalFrame.static.getNode_byLabel(prefix+"HO").m_local
        seg.setLength(2.0 * np.linalg.norm(top-bottom))

    def _thorax_motion(self,aqui, dictRef,dictAnat,options=None):

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment("Thorax")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]   # reinit Technical Frame Motion ()

        # additional markers
        # new markers
        valTop=(aqui.GetPoint("CLAV").GetValues() + aqui.GetPoint("C7").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,"midTop",valTop,desc="")

        valBottom=(aqui.GetPoint("STRN").GetValues() + aqui.GetPoint("T10").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,"midBottom",valBottom,desc="")

        valFront=(aqui.GetPoint("STRN").GetValues() + aqui.GetPoint("CLAV").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,"midFront",valFront,desc="")

        # computation
        LSJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))
        RSJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))
        LVWMvalues=np.zeros((aqui.GetPointFrameNumber(),3))
        RVWMvalues=np.zeros((aqui.GetPointFrameNumber(),3))
        OTvalues=np.zeros((aqui.GetPointFrameNumber(),3))

        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Thorax"]["TF"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

            OT = ptOrigin + -1.0*(markerDiameter/2.0)*csFrame.m_axisX

            LSHO = aqui.GetPoint(str("LSHO")).GetValues()[i,:]
            LVWM = np.cross((LSHO - OT ), csFrame.m_axisX ) + LSHO
            RSHO = aqui.GetPoint(str("RSHO")).GetValues()[i,:]
            RVWM = np.cross((RSHO - OT ), csFrame.m_axisX ) + RSHO

            if validFrames[i]:
                OTvalues[i,:] = OT
                LSJCvalues[i,:] = modelDecorator.VCMJointCentre( -1.0*(self.mp["LeftShoulderOffset"]+ markerDiameter/2.0) ,LSHO,OT,LVWM, beta=0 )
                LVWMvalues[i,:] = LVWM
                RSJCvalues[i,:] = modelDecorator.VCMJointCentre( 1.0*(self.mp["RightShoulderOffset"]+ markerDiameter/2.0) ,RSHO,OT,RVWM, beta=0 )
                RVWMvalues[i,:] = RVWM

        btkTools.smartAppendPoint(aqui,"OT",OTvalues,desc="")
        btkTools.smartAppendPoint(aqui,"LVWM",LVWMvalues,desc="")
        btkTools.smartAppendPoint(aqui,"RVWM",RVWMvalues,desc="")

        # --- LKJC
        if  "useLeftSJCmarker" in options.keys():
            LSJCvalues = aqui.GetPoint(options["useLeftSJCmarker"]).GetValues()
            desc = aqui.GetPoint(options["useLeftSJCmarker"]).GetDescription()
            btkTools.smartAppendPoint(aqui,"LSJC",LSJCvalues,desc=desc)
        else:
            desc = seg.getReferential('TF').static.getNode_byLabel("LSJC").m_desc
            btkTools.smartAppendPoint(aqui,"LSJC",LSJCvalues,desc=str("Chord-"+desc))

        if  "useRightSJCmarker" in options.keys():
            RSJCvalues = aqui.GetPoint(options["useRightSJCmarker"]).GetValues()
            desc = aqui.GetPoint(options["useRightSJCmarker"]).GetDescription()
            btkTools.smartAppendPoint(aqui,"RSJC",RSJCvalues,desc=desc)
        else:
            desc = seg.getReferential('TF').static.getNode_byLabel("RSJC").m_desc
            btkTools.smartAppendPoint(aqui,"RSJC",RSJCvalues,desc=str("Chord-"+desc))

        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]

        T5inThorax = np.zeros((aqui.GetPointFrameNumber(),3))
        C7inThorax = np.zeros((aqui.GetPointFrameNumber(),3))
        T10inThorax = np.zeros((aqui.GetPointFrameNumber(),3))

        # additional markers
        # NA
        # computation

        #self._TopLumbar5
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat["Thorax"]['labels'][0])).GetValues()[i,:] #midTop
            pt2=aqui.GetPoint(str(dictAnat["Thorax"]['labels'][1])).GetValues()[i,:] #midBottom
            pt3=aqui.GetPoint(str(dictAnat["Thorax"]['labels'][2])).GetValues()[i,:] #midFront
            ptOrigin=aqui.GetPoint(str(dictAnat["Thorax"]['labels'][3])).GetValues()[i,:] #OT


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictAnat["Thorax"]['sequence'])


            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))

            if hasattr(self,"_TopLumbar5"):
                T5inThorax[i,:] = np.dot(R.T,self._TopLumbar5[i,:]-ptOrigin)


            offset =( ptOrigin + np.dot(R,np.array([-markerDiameter/2.0,0,0])) - ptOrigin)*1.05

            C7Global= aqui.GetPoint(str("C7")).GetValues()[i,:] + offset
            C7inThorax[i,:] = np.dot(R.T,C7Global-ptOrigin)

            T10Global= aqui.GetPoint(str("T10")).GetValues()[i,:] + offset
            T10inThorax[i,:] = np.dot(R.T,T10Global-ptOrigin)


        if hasattr(self,"_TopLumbar5"):
            meanT5inThorax =np.mean(T5inThorax,axis=0)
            seg.anatomicalFrame.static.addNode("T5motion",meanT5inThorax,positionType="Local",desc = "meanTrial")

        meanC7inThorax =np.mean(C7inThorax,axis=0)
        meanT10inThorax =np.mean(T10inThorax,axis=0)


        seg.anatomicalFrame.static.addNode("C7motion",meanC7inThorax,positionType="Local",desc = "meanTrial")
        seg.anatomicalFrame.static.addNode("T10motion",meanT10inThorax,positionType="Local",desc = "meanTrial")

        if hasattr(self,"_TopLumbar5"):
            com = (meanC7inThorax + ( meanT5inThorax - meanC7inThorax ) * 0.63)
        else:
            com = (meanC7inThorax + ( meanT10inThorax - meanC7inThorax ) * 0.82)

        seg.anatomicalFrame.static.addNode("com",com,positionType="Local")



    def _clavicle_motion(self,side,aqui, dictRef,dictAnat,options=None):

        if side == "Left":
            prefix = "L"
        if side == "Right":
            prefix ="R"
            s= 1.0


        seg=self.getSegment(side+" Clavicle")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]   # reinit Technical Frame Motion ()

        # additional markers


        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef[side+" Clavicle"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef[side+" Clavicle"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef[side+" Clavicle"]["TF"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictRef[side+" Clavicle"]["TF"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef[side+" Clavicle"]["TF"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))


        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA
        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat[side+" Clavicle"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat[side+" Clavicle"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat[side+" Clavicle"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat[side+" Clavicle"]['labels'][3])).GetValues()[i,:]

            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictAnat[side+" Clavicle"]['sequence'])


            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))


    def _upperArm_motion(self,side,aqui, dictRef,dictAnat,options=None,frameReconstruction="Both"):

        if side == "Left":
            prefix = "L"
        if side == "Right":
            prefix ="R"
            s= 1.0

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment(side+" UpperArm")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        if frameReconstruction == "Both" or frameReconstruction == "Technical":
            # --- motion of the technical referential
            seg.getReferential("TF").motion =[]   # reinit Technical Frame Motion ()

            # additional markers

            # CVMvalues=np.zeros((aqui.GetPointFrameNumber(),3))
            # for i in range(0,aqui.GetPointFrameNumber()):
            #     SJC = aqui.GetPoint(prefix+"SJC").GetValues()[i,:]
            #     LHE=aqui.GetPoint(prefix+"ELB").GetValues()[i,:]
            #     MWP=aqui.GetPoint(prefix+"MWP").GetValues()[i,:]
            #
            #
            #     CVM = -1.0*np.cross((MWP-LHE),(SJC-LHE))
            #     CVM = CVM / np.linalg.norm(CVM)
            #     CVMvalues[i,:] =LHE + 50.0*CVM
            #
            # btkTools.smartAppendPoint(aqui,prefix+"CVM", CVMvalues, desc="")

            # computation
            EJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))

            csFrame=frame.Frame()
            for i in range(0,aqui.GetPointFrameNumber()):

                pt1=aqui.GetPoint(str(dictRef[side+" UpperArm"]["TF"]['labels'][0])).GetValues()[i,:]
                pt2=aqui.GetPoint(str(dictRef[side+" UpperArm"]["TF"]['labels'][1])).GetValues()[i,:]
                pt3=aqui.GetPoint(str(dictRef[side+" UpperArm"]["TF"]['labels'][2])).GetValues()[i,:]
                ptOrigin=aqui.GetPoint(str(dictRef[side+" UpperArm"]["TF"]['labels'][3])).GetValues()[i,:]


                a1=(pt2-pt1)
                a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

                v=(pt3-pt1)
                v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

                a2=np.cross(a1,v)
                a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

                x,y,z,R=frame.setFrameData(a1,a2,dictRef[side+" UpperArm"]["TF"]['sequence'])

                csFrame.m_axisX=x
                csFrame.m_axisY=y
                csFrame.m_axisZ=z
                csFrame.setRotation(R)
                csFrame.setTranslation(ptOrigin)

                seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

                SJC = aqui.GetPoint(prefix+"SJC").GetValues()[i,:]
                LHE=aqui.GetPoint(prefix+"ELB").GetValues()[i,:]
                CVM = aqui.GetPoint(prefix+"CVM").GetValues()[i,:]

                #EJCvalues[i,:] =  modelDecorator.VCMJointCentre( (self.mp[side+"ElbowWidth"]+ markerDiameter)/2.0 ,LHE,SJC,CVM, beta=0 )
                if validFrames[i]:
                    EJCvalues[i,:] =  modelDecorator.VCMJointCentre( (self.mp[side+"ElbowWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=0 )


            #btkTools.smartAppendPoint(aqui,"LKJC_Chord",LKJCvalues,desc="chord")

            # --- LKJC
            if  "useLeftEJCmarker" in options.keys():
                LEJCvalues = aqui.GetPoint(options["useLeftEJCmarker"]).GetValues()
                desc = aqui.GetPoint(options["useLeftEJCmarker"]).GetDescription()
                btkTools.smartAppendPoint(aqui,"LEJC",LEJCvalues,desc=desc)
            else:
                desc = seg.getReferential('TF').static.getNode_byLabel(prefix+"EJC").m_desc
                #LKJCvalues = aqui.GetPoint("LKJC_Chord").GetValues()
                btkTools.smartAppendPoint(aqui,prefix+"EJC",EJCvalues,desc=str("Chord-"+desc))


            if  "useRightEJCmarker" in options.keys():
                REJCvalues = aqui.GetPoint(options["useRightEJCmarker"]).GetValues()
                desc = aqui.GetPoint(options["useRightEJCmarker"]).GetDescription()
                btkTools.smartAppendPoint(aqui,"REJC",LEJCvalues,desc=desc)
            else:
                desc = seg.getReferential('TF').static.getNode_byLabel(prefix+"EJC").m_desc
                #LKJCvalues = aqui.GetPoint("LKJC_Chord").GetValues()
                btkTools.smartAppendPoint(aqui,prefix+"EJC",EJCvalues,desc=str("Chord-"+desc))


        if frameReconstruction == "Both" or frameReconstruction == "Anatomical":
            # --- motion of the anatomical referential
            seg.anatomicalFrame.motion=[]


            # additional markers
            # NA
            # computation
            csFrame=frame.Frame()
            for i in range(0,aqui.GetPointFrameNumber()):

                pt1=aqui.GetPoint(str(dictAnat[side+" UpperArm"]['labels'][0])).GetValues()[i,:]
                pt2=aqui.GetPoint(str(dictAnat[side+" UpperArm"]['labels'][1])).GetValues()[i,:]
                pt3=aqui.GetPoint(str(dictAnat[side+" UpperArm"]['labels'][2])).GetValues()[i,:]
                ptOrigin=aqui.GetPoint(str(dictAnat[side+" UpperArm"]['labels'][3])).GetValues()[i,:]


                a1=(pt2-pt1)
                a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

                v=(pt3-pt1)
                v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

                a2=np.cross(a1,v)
                a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

                x,y,z,R=frame.setFrameData(a1,a2,dictAnat[side+" UpperArm"]['sequence'])


                csFrame.m_axisX=x
                csFrame.m_axisY=y
                csFrame.m_axisZ=z
                csFrame.setRotation(R)
                csFrame.setTranslation(ptOrigin)

                seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))

    def _foreArm_motion(self,side,aqui, dictRef,dictAnat,options=None, frameReconstruction="both"):

        if side == "Left":
            prefix = "L"
            s = -1.0
        if side == "Right":
            prefix ="R"
            s= 1.0

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment(side+" ForeArm")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        if frameReconstruction == "Both" or frameReconstruction == "Technical":
        # --- motion of the technical referential
            seg.getReferential("TF").motion =[]   # reinit Technical Frame Motion ()

            # additional markers

            # computation
            WJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))

            csFrame=frame.Frame()
            for i in range(0,aqui.GetPointFrameNumber()):

                pt1=aqui.GetPoint(str(dictRef[side+" ForeArm"]["TF"]['labels'][0])).GetValues()[i,:]#
                pt2=aqui.GetPoint(str(dictRef[side+" ForeArm"]["TF"]['labels'][1])).GetValues()[i,:]
                pt3=aqui.GetPoint(str(dictRef[side+" ForeArm"]["TF"]['labels'][2])).GetValues()[i,:]
                ptOrigin=aqui.GetPoint(str(dictRef[side+" ForeArm"]["TF"]['labels'][3])).GetValues()[i,:]


                a1=(pt2-pt1)
                a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

                v=(pt3-pt1)
                v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

                a2=np.cross(a1,v)
                a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

                x,y,z,R=frame.setFrameData(a1,a2,dictRef[side+" ForeArm"]["TF"]['sequence'])

                csFrame.m_axisX=x
                csFrame.m_axisY=y
                csFrame.m_axisZ=z
                csFrame.setRotation(R)
                csFrame.setTranslation(ptOrigin)

                seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

                EJC = pt2
                US=pt3
                RS=pt1

                MWP=aqui.GetPoint(prefix+"MWP").GetValues()[i,:]

                WJCaxis = np.cross((US-RS),(EJC-MWP))
                WJCaxis = WJCaxis / np.linalg.norm(WJCaxis)
                if validFrames[i]:
                    WJCvalues[i,:] =MWP +  (s*(self.mp[side +"WristWidth"]+markerDiameter)/2.0)*WJCaxis


                EJC=aqui.GetPoint(prefix+"EJC").GetValues()[i,:]
                US=aqui.GetPoint(prefix+"WRB").GetValues()[i,:]
                RS=aqui.GetPoint(prefix+"WRA").GetValues()[i,:]
                MWP=aqui.GetPoint(prefix+"MWP").GetValues()[i,:]


            #btkTools.smartAppendPoint(aqui,"LKJC_Chord",LKJCvalues,desc="chord")

            if  "useLeftWJCmarker" in options.keys():
                LWJCvalues = aqui.GetPoint(options["useLeftWJCmarker"]).GetValues()
                desc = aqui.GetPoint(options["useLeftWJCmarker"]).GetDescription()
                btkTools.smartAppendPoint(aqui,"LWJC",LWJCvalues,desc=desc)
            else:
                desc = seg.getReferential('TF').static.getNode_byLabel(prefix+"WJC").m_desc
                #LKJCvalues = aqui.GetPoint("LKJC_Chord").GetValues()
                btkTools.smartAppendPoint(aqui,prefix+"WJC",WJCvalues,desc=str("Chord-"+desc))

            if  "useRightWJCmarker" in options.keys():
                RWJCvalues = aqui.GetPoint(options["useRightWJCmarker"]).GetValues()
                desc = aqui.GetPoint(options["useRightWJCmarker"]).GetDescription()
                btkTools.smartAppendPoint(aqui,"RWJC",RWJCvalues,desc=desc)
            else:
                desc = seg.getReferential('TF').static.getNode_byLabel(prefix+"WJC").m_desc
                #LKJCvalues = aqui.GetPoint("LKJC_Chord").GetValues()
                btkTools.smartAppendPoint(aqui,prefix+"WJC",WJCvalues,desc=str("Chord-"+desc))


        if frameReconstruction == "Both" or frameReconstruction == "Anatomical":
            # --- motion of the anatomical referential
            seg.anatomicalFrame.motion=[]


            # computation
            csFrame=frame.Frame()
            for i in range(0,aqui.GetPointFrameNumber()):

                pt1=aqui.GetPoint(str(dictAnat[side+" ForeArm"]['labels'][0])).GetValues()[i,:]
                pt2=aqui.GetPoint(str(dictAnat[side+" ForeArm"]['labels'][1])).GetValues()[i,:]
                if dictAnat[side+" ForeArm"]['labels'][2] is not None:
                    pt3=aqui.GetPoint(str(dictAnat[side+" ForeArm"]['labels'][2])).GetValues()[i,:]

                ptOrigin=aqui.GetPoint(str(dictAnat[side+" ForeArm"]['labels'][3])).GetValues()[i,:]


                a1=(pt2-pt1)
                a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

                if dictAnat[side+" ForeArm"]['labels'][2] is not None:
                    v=(pt3-pt1)
                    v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))
                else:
                    v=self.getSegment(side+" UpperArm").anatomicalFrame.motion[i].m_axisY

                a2=np.cross(a1,v)
                a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

                x,y,z,R=frame.setFrameData(a1,a2,dictAnat[side+" ForeArm"]['sequence'])

                csFrame.m_axisX=x
                csFrame.m_axisY=y
                csFrame.m_axisZ=z
                csFrame.setRotation(R)
                csFrame.setTranslation(ptOrigin)

                seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))

    def _hand_motion(self,side,aqui, dictRef,dictAnat,options=None):

        if side == "Left":
            prefix = "L"
        if side == "Right":
            prefix ="R"
            s= 1.0

        if "markerDiameter" in options.keys():
            LOGGER.logger.debug(" option (markerDiameter) found ")
            markerDiameter = options["markerDiameter"]
        else:
            markerDiameter=14.0

        if "basePlate" in options.keys():
            LOGGER.logger.debug(" option (basePlate) found ")
            basePlate = options["basePlate"]
        else:
            basePlate=2.0

        seg=self.getSegment(side+" Hand")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]   # reinit Technical Frame Motion ()

        # additional markers
        midwrist=(aqui.GetPoint(prefix+"WRA").GetValues() + aqui.GetPoint(prefix+"WRB").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,prefix+"MWP",midwrist,desc="")


        # computation
        HOvalues=np.zeros((aqui.GetPointFrameNumber(),3))

        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef[side+" Hand"]["TF"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictRef[side+" Hand"]["TF"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictRef[side+" Hand"]["TF"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictRef[side+" Hand"]["TF"]['labels'][3])).GetValues()[i,:]

            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef[side+" Hand"]["TF"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))

            WJC=aqui.GetPoint(prefix+"WJC").GetValues()[i,:]
            MH2=aqui.GetPoint(prefix+"FIN").GetValues()[i,:]
            MWP=aqui.GetPoint(prefix+"MWP").GetValues()[i,:]
            if validFrames[i]:
                HOvalues[i,:] =  modelDecorator.VCMJointCentre( (self.mp[side+"HandThickness"]+ markerDiameter)/2.0 ,MH2, WJC, MWP, beta=0 )


        if  "useLeftHOmarker" in options.keys():
            LHOvalues = aqui.GetPoint(options["useLeftHOmarker"]).GetValues()
            desc = aqui.GetPoint(options["useLeftHOmarker"]).GetDescription()
            btkTools.smartAppendPoint(aqui,"LHO",LHOvalues,desc=desc)
        else:
            desc = seg.getReferential('TF').static.getNode_byLabel(prefix+"HO").m_desc
            btkTools.smartAppendPoint(aqui,prefix+"HO",HOvalues,desc=str("Chord-"+desc))

        if  "useRightHOmarker" in options.keys():
            RHOvalues = aqui.GetPoint(options["useRightHOmarker"]).GetValues()
            desc = aqui.GetPoint(options["useRightHOmarker"]).GetDescription()
            btkTools.smartAppendPoint(aqui,"RHO",RHOvalues,desc=desc)
        else:
            desc = seg.getReferential('TF').static.getNode_byLabel(prefix+"HO").m_desc
            btkTools.smartAppendPoint(aqui,prefix+"HO",HOvalues,desc=str("Chord-"+desc))

        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA
        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictAnat[side+" Hand"]['labels'][0])).GetValues()[i,:]
            pt2=aqui.GetPoint(str(dictAnat[side+" Hand"]['labels'][1])).GetValues()[i,:]
            pt3=aqui.GetPoint(str(dictAnat[side+" Hand"]['labels'][2])).GetValues()[i,:]
            ptOrigin=aqui.GetPoint(str(dictAnat[side+" Hand"]['labels'][3])).GetValues()[i,:]


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictAnat[side+" Hand"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))

    def _head_motion(self,aqui, dictRef,dictAnat,options=None):

        seg=self.getSegment("Head")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        valmFH=(aqui.GetPoint("LFHD").GetValues() + aqui.GetPoint("RFHD").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,"midFH",valmFH,desc="")

        valmBH=(aqui.GetPoint("LBHD").GetValues() + aqui.GetPoint("RBHD").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,"midBH",valmBH,desc="")

        valmLH=(aqui.GetPoint("LFHD").GetValues() + aqui.GetPoint("LBHD").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,"midLH",valmLH,desc="")


        valmHC=(valmFH+valmBH) / 2.0
        btkTools.smartAppendPoint(aqui,"HC",valmHC,desc="")

        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):

            pt1=aqui.GetPoint(str(dictRef["Head"]["TF"]['labels'][0])).GetValues()[i,:] #toe
            pt2=aqui.GetPoint(str(dictRef["Head"]["TF"]['labels'][1])).GetValues()[i,:] #ajc
            pt3=aqui.GetPoint(str(dictRef["Head"]["TF"]['labels'][2])).GetValues()[i,:] #ajc
            ptOrigin=aqui.GetPoint(str(dictRef["Head"]["TF"]['labels'][3])).GetValues()[i,:]

            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef["Head"]["TF"]['sequence'])

            csFrame.m_axisX=x
            csFrame.m_axisY=y
            csFrame.m_axisZ=z
            csFrame.setRotation(R)
            csFrame.setTranslation(ptOrigin)

            seg.getReferential("TF").addMotionFrame(copy.deepcopy(csFrame))


        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA

        # computation
        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            ptOrigin=aqui.GetPoint(str(dictAnat["Head"]['labels'][3])).GetValues()[i,:]
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)

            csFrame.update(R,ptOrigin)
            seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))


    # --- opensim --------
    def opensimTrackingMarkers(self):

        excluded = ["Thorax","Head","Left Clavicle", "Left UpperArm", "Left ForeArm",
                    "Left Hand", "Right Clavicle", "Right UpperArm", "Right ForeArm",
                     "Right Hand"]


        out={}
        for segIt in self.m_segmentCollection:
            if not segIt.m_isCloneOf and segIt.name not in excluded:
                out[segIt.name] = segIt.m_tracking_markers

        return out



    def opensimGeometry(self):
        """
        return dict used to configure the osim file
        """

        out={}
        out["hip_r"]= {"joint label":"RHJC", "proximal segment label":"Pelvis", "distal segment label":"Right Thigh" }
        out["knee_r"]= {"joint label":"RKJC", "proximal segment label":"Right Thigh", "distal segment label":"Right Shank" }
        out["ankle_r"]= {"joint label":"RAJC", "proximal segment label":"Right Shank", "distal segment label":"Right Foot" }
        #out["mtp_r"]=


        out["hip_l"]= {"joint label":"LHJC", "proximal segment label":"Pelvis", "distal segment label":"Left Thigh" }
        out["knee_l"]= {"joint label":"LKJC", "proximal segment label":"Left Thigh", "distal segment label":"Left Shank" }
        out["ankle_l"]= {"joint label":"LAJC", "proximal segment label":"Left Shank", "distal segment label":"Left Foot" }
        #out["mtp_l"]=

        return out

    def opensimIkTask(self):
        """ return marker weights used for IK"""

        out={"LASI":100,
             "RASI":100,
             "LPSI":100,
             "RPSI":100,
             "RTHI":100,
             "RKNE":100,
             "RTIB":100,
             "RANK":100,
             "RHEE":100,
             "RTOE":100,
             "LTHI":100,
             "LKNE":100,
             "LTIB":100,
             "LANK":100,
             "LHEE":100,
             "LTOE":100,
             }

        return out



    # ----- vicon API -------
    def viconExport(self, NEXUS, acq, vskName, pointSuffix, staticProcessingFlag):
        """
        method exporting model outputs to Nexus

        Args:
            NEXUS (viconnexus): Nexus handle
            vskName (str): vsk name
            staticProcessingFlag (bool):  only static model ouputs will be exported

        """

        pointSuffix  =  pointSuffix if pointSuffix is not None else ""

        if staticProcessingFlag:
            if self.checkCalibrationProperty("LeftKAD",True):
                nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LKNE", acq)
            if self.checkCalibrationProperty("RightKAD",True):
                nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RKNE", acq)

        # export measured markers ( for CGM2.2 and 2.3)
        for it in btk.Iterate(acq.GetPoints()):
            if "_m" in it.GetLabel():
                nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,it.GetLabel(),acq)

        # export JC
        jointcentres = ["LHJC","RHJC","LKJC","RKJC","LAJC","RAJC","LSJC","RSJC","LEJC","REJC","LHO","RHO"]

        for jointCentre in jointcentres:
            if btkTools.isPointExist(acq, jointCentre):
                nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,jointCentre, acq,suffix = pointSuffix)

        LOGGER.logger.debug("jc over")

        # export angles
        for it in btk.Iterate(acq.GetPoints()):
            if it.GetType() == btk.btkPoint.Angle:
                if pointSuffix is not None:
                    if pointSuffix in it.GetLabel():
                        nexusTools.appendAngleFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                else:
                    nexusTools.appendAngleFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)

        LOGGER.logger.debug("angles over")

        # bones
        # -------------
        if btkTools.isPointExist(acq, "midHJC"):
            nexusTools.appendBones(NEXUS,vskName,acq,"PELVIS", self.getSegment("Pelvis"),
                OriginValues = acq.GetPoint("midHJC").GetValues(), suffix = pointSuffix, existFromPoint = "LPelvisAngles" )

        if btkTools.isPointExist(acq, "LKJC"):
            nexusTools.appendBones(NEXUS,vskName,acq,"LFEMUR", self.getSegment("Left Thigh"),
                OriginValues = acq.GetPoint("LKJC").GetValues(),suffix = pointSuffix , existFromPoint = "LHipAngles")
            #nexusTools.appendBones(NEXUS,vskName,"LFEP", self.getSegment("Left Shank Proximal"),OriginValues = acq.GetPoint("LKJC").GetValues(),manualScale = 100 )
        if btkTools.isPointExist(acq, "LAJC"):
            nexusTools.appendBones(NEXUS,vskName,acq,"LTIBIA", self.getSegment("Left Shank"),
                    OriginValues = acq.GetPoint("LAJC").GetValues(),suffix = pointSuffix ,existFromPoint = "LKneeAngles")

        if btkTools.isPointExist(acq, "LAJC") and  btkTools.isPointExist(acq,"LTOE",ignorePhantom=False):
            nexusTools.appendBones(NEXUS,vskName,acq,"LFOOT", self.getSegment("Left Foot"),
                OriginValues = self.getSegment("Left Foot").anatomicalFrame.getNodeTrajectory("FootOriginOffset"),suffix = pointSuffix, existFromPoint = "LAnkleAngles")
            nexusTools.appendBones(NEXUS,vskName,acq,"LTOES", self.getSegment("Left Foot"),
                OriginValues = self.getSegment("Left Foot").anatomicalFrame.getNodeTrajectory("ToeOrigin"),  manualScale = self.getSegment("Left Foot").m_bsp["length"]/3.0,suffix = pointSuffix, existFromPoint = "LAnkleAngles" )

        if btkTools.isPointExist(acq, "RKJC"):
            nexusTools.appendBones(NEXUS,vskName,acq,"RFEMUR", self.getSegment("Right Thigh"),
                OriginValues = acq.GetPoint("RKJC").GetValues(),suffix = pointSuffix, existFromPoint = "RHipAngles" )
                #nexusTools.appendBones(NEXUS,vskName,"RFEP", self.getSegment("Right Shank Proximal"),OriginValues = acq.GetPoint("RKJC").GetValues(),manualScale = 100 )
        if btkTools.isPointExist(acq, "RAJC"):
            nexusTools.appendBones(NEXUS,vskName,acq,"RTIBIA", self.getSegment("Right Shank"),
                OriginValues = acq.GetPoint("RAJC").GetValues() ,suffix = pointSuffix, existFromPoint = "RKneeAngles")

        if btkTools.isPointExist(acq, "RAJC") and  btkTools.isPointExist(acq,"RTOE",ignorePhantom=False):
            nexusTools.appendBones(NEXUS,vskName,acq,"RFOOT", self.getSegment("Right Foot") ,
                OriginValues = self.getSegment("Right Foot").anatomicalFrame.getNodeTrajectory("FootOriginOffset"),suffix = pointSuffix,existFromPoint = "RAnkleAngles" )
            nexusTools.appendBones(NEXUS,vskName,acq,"RTOES", self.getSegment("Right Foot") ,
                OriginValues = self.getSegment("Right Foot").anatomicalFrame.getNodeTrajectory("ToeOrigin"),
                manualScale = self.getSegment("Right Foot").m_bsp["length"]/3.0,suffix = pointSuffix, existFromPoint = "RAnkleAngles")

        if btkTools.isPointExist(acq, "OT"):

            nexusTools.appendBones(NEXUS,vskName,acq,"THORAX", self.getSegment("Thorax"),
                OriginValues = acq.GetPoint("OT").GetValues(),
                manualScale = self.getSegment("Thorax").m_info["Scale"],
                suffix = pointSuffix, existFromPoint = "LThoraxAngles" )

        if btkTools.isPointExist(acq, "LEJC"):
            nexusTools.appendBones(NEXUS,vskName,acq,"LUPPERARM", self.getSegment("Left UpperArm"),
                OriginValues = acq.GetPoint("LEJC").GetValues(),suffix = pointSuffix,existFromPoint = "LShoulderAngles" )

        if btkTools.isPointExist(acq, "LWJC"):
            nexusTools.appendBones(NEXUS,vskName,acq,"LFOREARM", self.getSegment("Left ForeArm"),
                OriginValues = acq.GetPoint("LWJC").GetValues(),suffix = pointSuffix,existFromPoint = "LElbowAngles" )

        if btkTools.isPointExist(acq, "LHO"):
            nexusTools.appendBones(NEXUS,vskName,acq,"LHAND", self.getSegment("Left Hand"),
                OriginValues = acq.GetPoint("LHO").GetValues(),suffix = pointSuffix,existFromPoint = "LWristAngles" )

        if btkTools.isPointExist(acq, "REJC"):
            nexusTools.appendBones(NEXUS,vskName,acq,"RUPPERARM", self.getSegment("Right UpperArm"),
                OriginValues = acq.GetPoint("REJC").GetValues(),suffix = pointSuffix, existFromPoint = "RShoulderAngles" )

        if btkTools.isPointExist(acq, "RWJC"):
            nexusTools.appendBones(NEXUS,vskName,acq,"RFOREARM", self.getSegment("Right ForeArm"),
                OriginValues = acq.GetPoint("RWJC").GetValues(),suffix = pointSuffix, existFromPoint = "RElbowAngles" )

        if btkTools.isPointExist(acq, "RHO"):
            nexusTools.appendBones(NEXUS,vskName,acq,"RHAND", self.getSegment("Right Hand"),
                OriginValues = acq.GetPoint("RHO").GetValues(),suffix = pointSuffix, existFromPoint = "RWristAngles" )

        nexusTools.appendBones(NEXUS,vskName,acq,"HEAD", self.getSegment("Head"),
            OriginValues = self.getSegment("Head").anatomicalFrame.getNodeTrajectory("SkullOriginOffset"),
            manualScale = self.getSegment("Head").m_info["headScale"],suffix = pointSuffix, existFromPoint = "LHeadAngles" )
        LOGGER.logger.debug("bones over")

        if not staticProcessingFlag:
            # export Force
            for it in btk.Iterate(acq.GetPoints()):
                if it.GetType() == btk.btkPoint.Force:
                    if pointSuffix is not None:
                        if pointSuffix in it.GetLabel():
                            nexusTools.appendForceFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                    else:
                        nexusTools.appendForceFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
            LOGGER.logger.debug("force over")

            # export Moment
            for it in btk.Iterate(acq.GetPoints()):
                if it.GetType() == btk.btkPoint.Moment:
                    if pointSuffix is not None:
                        if pointSuffix in it.GetLabel():
                            nexusTools.appendMomentFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                    else:
                        nexusTools.appendMomentFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
            LOGGER.logger.debug("Moment over")

            # export Moment
            for it in btk.Iterate(acq.GetPoints()):
                if it.GetType() == btk.btkPoint.Power:
                    if pointSuffix is not None:
                        if pointSuffix in it.GetLabel():
                            nexusTools.appendPowerFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
                    else:
                        nexusTools.appendPowerFromAcq(NEXUS,vskName,str(it.GetLabel()), acq)
            LOGGER.logger.debug("power over")

        # centre of mass
        centreOfMassLabel  = "CentreOfMass" + pointSuffix if pointSuffix is not None else "CentreOfMass"
        if btkTools.isPointExist(acq, centreOfMassLabel):
            nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,centreOfMassLabel, acq)
