# -*- coding: utf-8 -*-
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
import copy
from typing import List, Tuple, Dict, Optional,Union,Any

import btk

from pyCGM2 import enums
from pyCGM2.Model import model, modelDecorator, frame, motion
from pyCGM2.Math import euler
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusTools

import os
from pyCGM2.decorators.tracker import time_tracker #    @time_tracker()
import time



class CGM(model.Model):
    """
    Base Class of the Conventional Gait Model
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
                            'Left': ["LHipMoment","LKneeMoment","LAnkleMoment","LHipPower","LKneePower","LAnklePower","LStanGroundReactionForce","LGroundReactionForce"],
                            'Right': ["RHipMoment","RKneeMoment","RAnkleMoment","RHipPower","RKneePower","RAnklePower","RStanGroundReactionForce","RGroundReactionForce"]}

    VERSIONS = ["CGM1", "CGM1.1", "CGM2.1",  "CGM2.2", "CGM2.3", "CGM2.4", "CGM2.5"]


    def __init__(self):

        super(CGM, self).__init__()
        self.m_useLeftTibialTorsion=False
        self.m_useRightTibialTorsion=False
        self.staExpert= False

        self.m_staticTrackingMarkers = None

    def setSTAexpertMode(self, boolFlag: bool) -> None:
        """
        Set STA expert mode.

        Args:
            boolFlag (bool): Flag indicating whether to enable STA expert mode.

        """
        self.staExpert= boolFlag

    def setStaticTrackingMarkers(self, markers: List[str]) -> None:
        """
        Set tracking markers.

        Args:
            markers (List[str]): List of tracking markers.

        """


        self.m_staticTrackingMarkers = markers

    def getStaticTrackingMarkers(self) -> Union[None, List[str]]:
        """
        Get tracking markers.

        Returns:
            Union[None, List[str]]: List of tracking markers or None if not set.

        """

        return self.m_staticTrackingMarkers

    @classmethod
    def detectCalibrationMethods(cls, acqStatic: btk.btkAcquisition)-> Dict[str, enums.JointCalibrationMethod]:
        """
        *Class method* to detect the method used to calibrate knee and ankle joint centres.

        Args:
            acqStatic (btk.btkAcquisition): btkAcquisition.

        Returns:
            Dict[str, JointCalibrationMethod]: Dictionary mapping joint names to calibration methods.

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

        dectectedCalibrationMethods={}
        dectectedCalibrationMethods["Left Knee"] = LKnee
        dectectedCalibrationMethods["Right Knee"] = RKnee
        dectectedCalibrationMethods["Left Ankle"] = LAnkle
        dectectedCalibrationMethods["Right Ankle"] = RAnkle

        return dectectedCalibrationMethods

    @classmethod
    def get_markerLabelForPiGStatic(cls, dcm: Dict[str, enums.JointCalibrationMethod]) -> List[str]:
        """
        *Class method* returning marker labels of the knee and ankle joint centres.

        Args:
            dcm (Dict[str, enums.JointCalibrationMethod]): Dictionary returned from the function `detectCalibrationMethods`.

        Returns:
            List[str]: List of marker labels for knee and ankle joint centres.

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
    
    def _frameByFrameAxesConstruction(self, pt1,pt2,pt3,v=None):


        differences_21 = pt2 - pt1  
        if pt3 is not None: differences_31 = pt3 - pt1
        norms_21 = np.linalg.norm(differences_21, axis=1)
        if pt3 is not None:norms_31 = np.linalg.norm(differences_31, axis=1)

        a1 = np.nan_to_num(differences_21 / norms_21[:, np.newaxis])
        if pt3 is not None:
            v1 = np.nan_to_num(differences_31 / norms_31[:, np.newaxis])
        else:
            v1=v

        a2 = np.cross(a1, v1)
        a2 = np.nan_to_num(a2 / np.linalg.norm(a2, axis=1)[:, np.newaxis])
        

        return a1,a2


class CGM1(CGM):
    """
    Conventional Gait Model 1 (aka Vicon Plugin Gait Clone).
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

    def setVersion(self, string: str) -> None:
        """
        Amend the model version.

        Args:
            string (str): New version string to be set.
        """
        self.version = string

    def __repr__(self):
        return "CGM1.0"

    def _lowerLimbTrackingMarkers(self):
        """Get lower limb tracking markers.

        Returns:
            List[str]: A list of lower limb tracking markers.
        """
        return CGM1.LOWERLIMB_TRACKING_MARKERS#["LASI", "RASI","RPSI", "LPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

    def _trunkTrackingMarkers(self):
        """Get trunk tracking markers.

        Returns:
            List[str]: A list of lower limb tracking markers.
        """
        return CGM1.THORAX_TRACKING_MARKERS#["C7", "T10","CLAV", "STRN"]

    def _upperLimbTrackingMarkers(self):
        """Get upper limb tracking markers.

        Returns:
            List[str]: A list of lower limb tracking markers.
        """
        return CGM1.THORAX_TRACKING_MARKERS+CGM1.UPPERLIMB_TRACKING_MARKERS#S#["C7", "T10","CLAV", "STRN", "LELB", "LWRA", "LWRB", "LFRM", "LFIN", "RELB", "RWRA", "RWRB", "RFRM", "RFIN"]


    def getTrackingMarkers(self, acq: btk.btkAcquisition) -> List[str]:
        """
        Return tracking markers.

        Args:
            acq (btk.btkAcquisition): Acquisition object.

        Returns:
            List[str]: List of tracking markers.
        """

        tracking_markers =  self._lowerLimbTrackingMarkers() + self._upperLimbTrackingMarkers()

        return tracking_markers

    def getStaticMarkers(self, dcm: Dict):
        """Return static markers based on the detected calibration methods.

        Args:
            dcm (Dict): Dictionary returned from the function `detectCalibrationMethods`.

        Returns:
            List[str]: List of static markers.
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


    def configure(self, detectedCalibrationMethods: Optional[Dict] = None):
        """Configure the model based on detected calibration methods.

        Args:
            detectedCalibrationMethods (Dict, optional): Dictionary of detected calibration methods. Defaults to None.
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
        """Configure lower limb segments and joints in the model.

        This method adds segments and chains for the lower limb, including pelvis, thighs, shanks, and feet. It also defines joints like hip, knee, and ankle for both left and right sides. The configuration is based on tracking markers and calibration markers. For some segments, a clone is created with a modified anatomical frame based on tibial rotation values.
        """
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
        """Configure the trunk segment and joints in the model.

        This method adds the thorax segment and defines the left and right spine joints connecting the thorax to the pelvis. Clinical descriptors for these segments are also set.
        """
        self.addSegment("Thorax",0,enums.SegmentSide.Central,calibration_markers=[], tracking_markers = ["CLAV","C7","T10","STRN"])

        self.addJoint("LSpine","Thorax","Pelvis", "YXZ","LSJC")
        self.addJoint("RSpine","Thorax","Pelvis", "YXZ","LSJC")

        self.setClinicalDescriptor("LSpine",enums.DataType.Angle, [0,1,2],[1.0,-1.0,-1.0], [np.radians(-180),0.0,np.radians(180)])
        self.setClinicalDescriptor("RSpine",enums.DataType.Angle, [0,1,2],[1.0,1.0,1.0], [np.radians(-180),0.0,np.radians(180)])
        self.setClinicalDescriptor("LThorax",enums.DataType.Angle,[0,1,2],[1.0,-1.0,1.0], [-np.radians(180),0.0,-np.radians(180)])
        self.setClinicalDescriptor("RThorax",enums.DataType.Angle,[0,1,2],[1.0,1.0,-1.0], [-np.radians(180),0.0,-np.radians(180)])

    def _upperLimbConfigure(self):
        """Configure upper limb segments and joints in the model.

        This method adds segments for the head, clavicles, upper arms, forearms, and hands for both left and right sides. Joints such as shoulder, elbow, wrist, and neck are defined along with their respective clinical descriptors.
        """
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
        """Define the calibration procedure for the model.

        This method outlines the calibration procedure by defining technical and anatomical coordinate systems for various segments. It involves lower limb, trunk, and upper limb calibration procedures.

        Returns:
            tuple: A tuple containing two dictionaries, one for the markers and sequence used for building the technical coordinate system and another for the anatomical coordinate system.
        """

        dictRef={}
        dictRefAnatomical={}

        self._lowerLimbCalibrationProcedure(dictRef,dictRefAnatomical)
        self._trunkCalibrationProcedure(dictRef,dictRefAnatomical)
        self._upperLimbCalibrationProcedure(dictRef,dictRefAnatomical)

        return dictRef,dictRefAnatomical

    def _lowerLimbCalibrationProcedure(self, dictRef: Dict[str, Any], dictRefAnatomical: Dict[str, Any]) -> None:
        """Define the lower limb calibration procedure.

        This method sets up the calibration process for lower limb segments including pelvis, thighs, shanks, and feet. It populates the provided dictionaries with the necessary information for technical and anatomical coordinate systems.

        Args:
            dictRef (Dict[str, Any]): A dictionary to store reference markers and sequences for technical coordinate system calibration.
            dictRefAnatomical (Dict[str, Any]): A dictionary to store reference markers and sequences for anatomical coordinate system calibration.
        """

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

    def _trunkCalibrationProcedure(self, dictRef: Dict[str, Any], dictRefAnatomical: Dict[str, Any]) -> None:
        """Define the trunk calibration procedure.

        This method sets up the calibration process for the trunk segment. It includes populating the provided dictionaries with markers and sequences for technical and anatomical coordinate systems related to the thorax.

        Args:
            dictRef (Dict[str, Any]): A dictionary to store reference markers and sequences for technical coordinate system calibration.
            dictRefAnatomical (Dict[str, Any]): A dictionary to store reference markers and sequences for anatomical coordinate system calibration.
        """

        dictRef["Thorax"]={"TF" : {'sequence':"ZYX", 'labels':   ["midTop","midBottom","midFront","CLAV"]} }
        dictRefAnatomical["Thorax"]= {'sequence':"ZYX", 'labels':  ["midTop","midBottom","midFront","OT"]}

    def _upperLimbCalibrationProcedure(self, dictRef: Dict[str, Any], dictRefAnatomical: Dict[str, Any]) -> None:
        """Define the upper limb calibration procedure.

        This method outlines the calibration procedure for upper limb segments such as clavicles, upper arms, forearms, and hands. It involves populating the given dictionaries with necessary data for both technical and anatomical coordinate systems.

        Args:
            dictRef (Dict[str, Any]): A dictionary to store reference markers and sequences for technical coordinate system calibration.
            dictRefAnatomical (Dict[str, Any]): A dictionary to store reference markers and sequences for anatomical coordinate system calibration.
        """


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
        dictRefAnatomical["Right Hand"]={'sequence':"ZYX", 'labels':   ["RHO","RWJC","RMWP","RWJC"]}

    def _lowerLimbCoordinateSystemDefinitions(self) -> None:
        """Define the coordinate system for lower limb segments.

        This method sets the coordinate system definitions for various lower limb segments such as pelvis, thighs, shanks, and feet for both left and right sides.
        """

        self.setCoordinateSystemDefinition( "Pelvis", "PELVIS", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Thigh", "LFEMUR", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Thigh", "RFEMUR", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Shank", "LTIBIA", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Shank", "RTIBIA", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Shank Proximal", "LTIBIAPROX", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Shank Proximal", "RTIBIAPROX", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Foot", "LFOOT", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Foot", "RFOOT", "Anatomic")

    def _trunkCoordinateSystemDefinitions(self) -> None:
        """Define the coordinate system for the trunk segment.

        This method establishes the coordinate system definition for the thorax segment.
        """
        self.setCoordinateSystemDefinition( "Thorax", "THORAX", "Anatomic")

    def _upperLimbCoordinateSystemDefinitions(self) -> None:
        """Define the coordinate system for upper limb segments.

        This method sets the coordinate system definitions for upper limb segments including clavicles, arms, forearms, hands, and head for both left and right sides.
        """

        self.setCoordinateSystemDefinition( "Left Clavicle", "LCLAVICLE", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Clavicle", "RCLAVICLE", "Anatomic")
        self.setCoordinateSystemDefinition( "Head", "HEAD", "Anatomic")
        self.setCoordinateSystemDefinition( "Left UpperArm", "LUPPERARM", "Anatomic")
        self.setCoordinateSystemDefinition( "Left ForeArm", "LFOREARM", "Anatomic")
        self.setCoordinateSystemDefinition( "Left Hand", "LHANDARM", "Anatomic")
        self.setCoordinateSystemDefinition( "Right UpperArm", "RUPPERARM", "Anatomic")
        self.setCoordinateSystemDefinition( "Right ForeArm", "RFOREARM", "Anatomic")
        self.setCoordinateSystemDefinition( "Right Hand", "RHANDARM", "Anatomic")


    def _coordinateSystemDefinitions(self) -> None:
        """Define the coordinate systems for all body segments.

        This comprehensive method combines the definition of coordinate systems for the lower limbs, trunk, and upper limbs.
        """

        self._lowerLimbCoordinateSystemDefinitions()
        self._trunkCoordinateSystemDefinitions()
        self._upperLimbCoordinateSystemDefinitions()

    def calibrate(self, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], dictAnatomic: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrate the model using static acquisition data.

        This method carries out the calibration of the model using static acquisition data, reference markers, and anatomical markers. It allows for optional parameters to customize the calibration process.

        Args:
            aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
            dictRef (Dict[str, Any]): Dictionary for technical coordinate system reference markers.
            dictAnatomic (Dict[str, Any]): Dictionary for anatomical coordinate system reference markers.
            options (Optional[Dict[str, Any]]): Additional options for calibration, if any.
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

        self._thigh_calibrate("Left",aquiStatic, dictRef,frameInit,frameEnd,options=options)
        self._thigh_calibrate("Right",aquiStatic, dictRef,frameInit,frameEnd,options=options)

        self._shank_calibrate("Left",aquiStatic, dictRef,frameInit,frameEnd,options=options)
        self._shank_calibrate("Right",aquiStatic, dictRef,frameInit,frameEnd,options=options)

        # calibration of anatomical Referentials
        self._pelvis_Anatomicalcalibrate(aquiStatic, dictAnatomic,frameInit,frameEnd)

        self._thigh_Anatomicalcalibrate("Left",aquiStatic, dictAnatomic,frameInit,frameEnd)
        self._thigh_Anatomicalcalibrate("Right",aquiStatic, dictAnatomic,frameInit,frameEnd)


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




        self._shank_Anatomicalcalibrate("Left",aquiStatic, dictAnatomic,frameInit,frameEnd)
        self._shank_Anatomicalcalibrate("Right",aquiStatic, dictAnatomic,frameInit,frameEnd)

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
        self._shankProximal_AnatomicalCalibrate("Left",aquiStatic,dictAnatomic,frameInit,frameEnd,options=options) # alter static Frame


        self.updateSegmentFromCopy("Right Shank Proximal", self.getSegment("Right Shank"))
        self._shankProximal_AnatomicalCalibrate("Right",aquiStatic,dictAnatomic,frameInit,frameEnd,options=options) # alter static Frame

        # ---- FOOT CALIBRATION
        #-------------------------------------
        # foot ( need  Y-axis of the shank anatomic Frame)
        self._unCorrectedFoot_calibrate("Left",aquiStatic, dictRef,frameInit,frameEnd,options=options)

        self._foot_corrected_calibrate("Left",aquiStatic, dictAnatomic,frameInit,frameEnd,options=options)

        self._unCorrectedFoot_calibrate("Right",aquiStatic, dictRef,frameInit,frameEnd,options=options)

        self._foot_corrected_calibrate("Right",aquiStatic, dictAnatomic,frameInit,frameEnd,options=options)

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
    def _pelvis_calibrate(self, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrate the pelvis segment using static acquisition data.

        This method calculates various pelvis-related parameters and constructs the technical referential for the pelvis segment based on the provided static acquisition data.

        Args:
            aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
            dictRef (Dict[str, Any]): Dictionary for reference markers used in technical coordinate system calibration.
            frameInit (int): The initial frame for calibration.
            frameEnd (int): The final frame for calibration.
            options (Optional[Dict[str, Any]]): Additional options for calibration, if any.
        """


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

        val=(aquiStatic.GetPoint("LASI").GetValues() + aquiStatic.GetPoint("RASI").GetValues()+aquiStatic.GetPoint("LPSI").GetValues() + aquiStatic.GetPoint("RPSI").GetValues()) / 4.0
        btkTools.smartAppendPoint(aquiStatic,"pelvisCentre",val, desc="")

        seg.addCalibrationMarkerLabel("SACR")
        seg.addCalibrationMarkerLabel("midASIS")
        seg.addCalibrationMarkerLabel("pelvisCentre")


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



    def _thigh_calibrate(self,side:str, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrate the thigh segment.

        Constructs the technical referential for the thigh segment and calculates the knee joint center based on static acquisition data.

        Args:
            side (str): bodyside
            aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
            dictRef (Dict[str, Any]): Dictionary for reference markers used in technical coordinate system calibration.
            frameInit (int): The initial frame for calibration.
            frameEnd (int): The final frame for calibration.
            options (Optional[Dict[str, Any]]): Additional options for calibration, if any.
        """

        prefix = "L" if side == "Left" else "R" if side == "Right" else ""
        

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

        seg = self.getSegment(f"{side} Thigh")
        seg.resetMarkerLabels()

        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef[f"{side} Thigh"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#KNE
        pt2=aquiStatic.GetPoint(str(dictRef[f"{side} Thigh"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#HJC
        pt3=aquiStatic.GetPoint(str(dictRef[f"{side} Thigh"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#THI

        ptOrigin=aquiStatic.GetPoint(str(dictRef[f"{side} Thigh"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef[f"{side} Thigh"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)

        # --- knee Joint centers location from chord method
        if f"{side}ThighRotation" in self.mp and self.mp[f"{side}ThighRotation"] != 0:
            LOGGER.logger.debug(f"{side}ThighRotation defined from your vsk file")
            self.mp_computed[f"{side}ThighRotationOffset"] = self.mp[f"{side}ThighRotation"]
        else:
            self.mp_computed[f"{side}ThighRotationOffset"] = 0.0

        if side == "Left": offset = -self.mp_computed[f"LeftThighRotationOffset"]
        if side == "Right": offset = self.mp_computed[f"RightThighRotationOffset"]

        KJC = modelDecorator.VCMJointCentre( (self.mp[f"{side}KneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta= offset)

        if tf.static.isNodeExist(f"{prefix}KJC"):
            nodeKJC = tf.static.getNode_byLabel(f"{prefix}KJC")
        else:
            tf.static.addNode(f"{prefix}KJC_chord",KJC,positionType="Global",desc = "Chord")
            tf.static.addNode(f"{prefix}KJC",KJC,positionType="Global",desc = "Chord")
            nodeKJC = tf.static.getNode_byLabel(f"{prefix}KJC")

        btkTools.smartAppendPoint(aquiStatic,f"{prefix}KJC",
                    nodeKJC.m_global* np.ones((pfn,3)),
                    desc=nodeKJC.m_desc)

        # node for all markers
        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        # for label in seg.m_calibration_markers:
        #     globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        #     tf.static.addNode(label,globalPosition,positionType="Global")

        node_prox = self.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel(f"{prefix}HJC")
        tf.static.addNode(f"{prefix}HJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

        #seg.addTrackingMarkerLabel("LHJC")
        #seg.addCalibrationMarkerLabel("LKJC")


    # def _right_thigh_calibrate(self, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
    #     """Calibrate the right thigh segment.

    #     Constructs the technical referential for the right thigh segment and calculates the knee joint center based on static acquisition data.

    #     Args:
    #         aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
    #         dictRef (Dict[str, Any]): Dictionary for reference markers used in technical coordinate system calibration.
    #         frameInit (int): The initial frame for calibration.
    #         frameEnd (int): The final frame for calibration.
    #         options (Optional[Dict[str, Any]]): Additional options for calibration, if any.
    #     """


    #     pfn = aquiStatic.GetPointFrameNumber()

    #     if "markerDiameter" in options.keys():
    #         LOGGER.logger.debug(" option (markerDiameter) found ")
    #         markerDiameter = options["markerDiameter"]
    #     else:
    #         markerDiameter=14.0

    #     if "basePlate" in options.keys():
    #         LOGGER.logger.debug(" option (basePlate) found ")
    #         basePlate = options["basePlate"]
    #     else:
    #         basePlate=2.0

    #     seg = self.getSegment("Right Thigh")
    #     seg.resetMarkerLabels()

    #     # --- Construction of the technical referential
    #     tf=seg.getReferential("TF")

    #     pt1=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #     pt2=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #     pt3=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

    #     ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Thigh"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

    #     a1=(pt2-pt1)
    #     a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

    #     v=(pt3-pt1)
    #     v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

    #     a2=np.cross(a1,v)
    #     a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

    #     x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right Thigh"]["TF"]['sequence'])

    #     tf.static.m_axisX=x
    #     tf.static.m_axisY=y
    #     tf.static.m_axisZ=z
    #     tf.static.setRotation(R)
    #     tf.static.setTranslation(ptOrigin)


    #     # --- knee Joint centers location
    #     if "RightThighRotation" in self.mp and self.mp["RightThighRotation"] != 0:
    #         LOGGER.logger.debug("RightThighRotation defined from your vsk file")
    #         self.mp_computed["RightThighRotationOffset"] = self.mp["RightThighRotation"]
    #     else:
    #         self.mp_computed["RightThighRotationOffset"] = 0.0

    #     RKJC = modelDecorator.VCMJointCentre( (self.mp["RightKneeWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3,beta=self.mp_computed["RightThighRotationOffset"] ) # could consider a previous offset

    #     if tf.static.isNodeExist("RKJC"):
    #         nodeRKJC = tf.static.getNode_byLabel("RKJC")
    #     else:
    #         tf.static.addNode("RKJC_chord",RKJC,positionType="Global",desc = "Chord")
    #         tf.static.addNode("RKJC",RKJC,positionType="Global",desc = "Chord")
    #         nodeRKJC = tf.static.getNode_byLabel("RKJC")

    #     btkTools.smartAppendPoint(aquiStatic,"RKJC",
    #                 nodeRKJC.m_global* np.ones((pfn,3)),
    #                 desc=nodeRKJC.m_desc)


    #     for label in seg.m_tracking_markers:
    #         globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #         tf.static.addNode(label,globalPosition,positionType="Global")

    #     # for label in seg.m_calibration_markers:
    #     #     globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #     #     tf.static.addNode(label,globalPosition,positionType="Global")

    #     node_prox = self.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC")
    #     tf.static.addNode("RHJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

    #     # seg.addTrackingMarkerLabel("RHJC")
    #     # seg.addCalibrationMarkerLabel("RKJC")

    def _shank_calibrate(self,side:str, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrate the shank segment.

        Constructs the technical referential for the shank segment and calculates the ankle joint center based on static acquisition data.

        Args:
            side (str): body side
            aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
            dictRef (Dict[str, Any]): Dictionary for reference markers used in technical coordinate system calibration.
            frameInit (int): The initial frame for calibration.
            frameEnd (int): The final frame for calibration.
            options (Optional[Dict[str, Any]]): Additional options for calibration, if any.
        """
        prefix = "L" if side == "Left" else "R" if side == "Right" else ""

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


        seg = self.getSegment(f"{side} Shank")
        seg.resetMarkerLabels()

        # ---  additional markers and Update of the marker segment list


        # --- Construction of the technical referential
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef[f"{side} Shank"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictRef[f"{side} Shank"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictRef[f"{side} Shank"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictRef[f"{side} Shank"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictRef[f"{side} Shank"]["TF"]['sequence'])

        tf.static.m_axisX=x
        tf.static.m_axisY=y
        tf.static.m_axisZ=z
        tf.static.setRotation(R)
        tf.static.setTranslation(ptOrigin)


        # --- ankle Joint centers location
        if f"{side}ShankRotation" in self.mp and self.mp[f"{side}ShankRotation"] != 0:
            LOGGER.logger.debug(f"{side}ShankRotation defined from your vsk file")
            self.mp_computed[f"{side}ShankRotationOffset"] = self.mp[f"{side}ShankRotation"]
        else:
            self.mp_computed[f"{side}ShankRotationOffset"]=0.0

        if side == "Left": offset = -self.mp_computed["LeftShankRotationOffset"]
        if side == "Right": offset = self.mp_computed["RightShankRotationOffset"]


        AJC = modelDecorator.VCMJointCentre( (self.mp[f"{side}AnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta= offset)

        # --- node manager
        if tf.static.isNodeExist(f"{prefix}AJC"):
            nodeAJC = tf.static.getNode_byLabel(f"{prefix}AJC")
        else:
            tf.static.addNode(f"{prefix}AJC_chord",AJC,positionType="Global",desc = "Chord")
            tf.static.addNode(f"{prefix}AJC",AJC,positionType="Global",desc = "Chord")
            nodeAJC = tf.static.getNode_byLabel(f"{prefix}AJC")

        btkTools.smartAppendPoint(aquiStatic,f"{prefix}AJC",
                    nodeAJC.m_global* np.ones((pfn,3)),
                    desc=nodeAJC.m_desc)


        for label in seg.m_tracking_markers:
            globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            tf.static.addNode(label,globalPosition,positionType="Global")

        node_prox = self.getSegment(f"{side} Thigh").getReferential("TF").static.getNode_byLabel(f"{prefix}KJC")
        tf.static.addNode(f"{prefix}KJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)


    # def _right_shank_calibrate(self, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
    #     """Calibrate the right shank segment.

    #     Constructs the technical referential for the right shank segment and calculates the ankle joint center based on static acquisition data.

    #     Args:
    #         aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
    #         dictRef (Dict[str, Any]): Dictionary for reference markers used in technical coordinate system calibration.
    #         frameInit (int): The initial frame for calibration.
    #         frameEnd (int): The final frame for calibration.
    #         options (Optional[Dict[str, Any]]): Additional options for calibration, if any.
    #     """


    #     pfn = aquiStatic.GetPointFrameNumber()

    #     if "markerDiameter" in options.keys():
    #         LOGGER.logger.debug(" option (markerDiameter) found ")
    #         markerDiameter = options["markerDiameter"]
    #     else:
    #         markerDiameter=14.0

    #     if "basePlate" in options.keys():
    #         LOGGER.logger.debug(" option (basePlate) found ")
    #         basePlate = options["basePlate"]
    #     else:
    #         basePlate=2.0



    #     seg = self.getSegment("Right Shank")
    #     seg.resetMarkerLabels()


    #     # --- Construction of the technical Referential
    #     tf=seg.getReferential("TF")

    #     pt1=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #     pt2=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #     pt3=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

    #     ptOrigin=aquiStatic.GetPoint(str(dictRef["Right Shank"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

    #     a1=(pt2-pt1)
    #     a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

    #     v=(pt3-pt1)
    #     v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

    #     a2=np.cross(a1,v)
    #     a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

    #     x,y,z,R=frame.setFrameData(a1,a2,dictRef["Right Shank"]["TF"]['sequence'])

    #     tf.static.m_axisX=x
    #     tf.static.m_axisY=y
    #     tf.static.m_axisZ=z
    #     tf.static.setRotation(R)
    #     tf.static.setTranslation(ptOrigin)

    #     # --- ankle Joint centers location
    #     if "RightShankRotation" in self.mp and self.mp["RightShankRotation"] != 0:
    #         LOGGER.logger.debug("RightShankRotation defined from your vsk file")
    #         self.mp_computed["RightShankRotationOffset"] = self.mp["RightShankRotation"]
    #     else:
    #         self.mp_computed["RightShankRotationOffset"]=0.0

    #     RAJC = modelDecorator.VCMJointCentre( (self.mp["RightAnkleWidth"]+ markerDiameter)/2.0 ,pt1,pt2,pt3, beta=self.mp_computed["RightShankRotationOffset"] )

    #     # --- node manager
    #     if tf.static.isNodeExist("RAJC"):
    #         nodeRAJC = tf.static.getNode_byLabel("RAJC")
    #     else:
    #         tf.static.addNode("RAJC_chord",RAJC,positionType="Global",desc = "Chord")
    #         tf.static.addNode("RAJC",RAJC,positionType="Global",desc = "Chord")
    #         nodeRAJC = tf.static.getNode_byLabel("RAJC")

    #     btkTools.smartAppendPoint(aquiStatic,"RAJC",
    #                 nodeRAJC.m_global* np.ones((pfn,3)),
    #                 desc=nodeRAJC.m_desc)


    #     for label in seg.m_tracking_markers:
    #         globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #         tf.static.addNode(label,globalPosition,positionType="Global")

    #     # for label in seg.m_calibration_markers:
    #     #     globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #     #     tf.static.addNode(label,globalPosition,positionType="Global")

    #     node_prox = self.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC")
    #     tf.static.addNode("RKJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

    #     # seg.addTrackingMarkerLabel("RKJC")
    #     # seg.addCalibrationMarkerLabel("RAJC")

    def _unCorrectedFoot_calibrate(self,side: str, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrate the uncorrected foot segment.

        Constructs the technical referential for the uncorrected foot segment based on static acquisition data.

        Args:
            side (str): body side
            aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
            dictRef (Dict[str, Any]): Dictionary for reference markers used in technical coordinate system calibration.
            frameInit (int): The initial frame for calibration.
            frameEnd (int): The final frame for calibration.
            options (Optional[Dict[str, Any]]): Additional options for calibration, if any.
        """

        prefix = "L" if side == "Left" else "R" if side == "Right" else ""

        seg = self.getSegment(f"{side} Foot")
        #seg.resetMarkerLabels()

        # ---  additional markers and Update of the marker segment list
        #seg.addMarkerLabel("LKJC") !!!

        if "useBodyBuilderFoot" in options.keys() and options["useBodyBuilderFoot"]:
            LOGGER.logger.debug("You use a Left uncorrected foot sequence different than native CGM1")
            dictRef[f"{side} Foot"]={"TF" : {'sequence':"ZYX", 'labels':   [f"{prefix}TOE",f"{prefix}AJC",f"{prefix}KJC",f"{prefix}AJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis


        # --- Construction of the technical Referential
        tf=seg.getReferential("TF")

        pt1=aquiStatic.GetPoint(str(dictRef[f"{side} Foot"]["TF"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#LTOE
        pt2=aquiStatic.GetPoint(str(dictRef[f"{side} Foot"]["TF"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)#AJC

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        if dictRef[f"{side} Foot"]["TF"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictRef[f"{side} Foot"]["TF"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)

            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))
            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))
            x,y,z,R=frame.setFrameData(a1,a2,dictRef[f"{side} Foot"]["TF"]['sequence'])

        else:
            distalShank = self.getSegment(f"{side} Shank")
            proximalShank = self.getSegment(f"{side} Shank Proximal")

            # uncorrected Refrence with dist shank
            v=distalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))
            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))
            x_dist,y_dist,z_dist,R_dist=frame.setFrameData(a1,a2,dictRef[f"{side} Foot"]["TF"]['sequence'])

            # uncorrected Refrence with prox shank
            v=proximalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))
            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))
            x_prox,y_prox,z_prox,R_prox=frame.setFrameData(a1,a2,dictRef[f"{side} Foot"]["TF"]['sequence'])


            if side == "Left": self._R_leftUnCorrfoot_dist_prox = np.dot(R_prox.T,R_dist) # will be used for placing the foot uncorrected RF
            if side == "Right": self._R_rightUnCorrfoot_dist_prox = np.dot(R_prox.T,R_dist)

            if "viconCGM1compatible" in options.keys() and options["viconCGM1compatible"]:
                if distalShank.getReferential("TF").static.getNode_byLabel(f"{prefix}AJC").m_desc != "mid":
                    x,y,z,R = x_prox,y_prox,z_prox,R_prox
                else:
                    x,y,z,R = x_dist,y_dist,z_dist,R_dist
            else:
                x,y,z,R = x_dist,y_dist,z_dist,R_dist

        ptOrigin=aquiStatic.GetPoint(str(dictRef[f"{side} Foot"]["TF"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

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

        node_prox = self.getSegment(f"{side} Shank").getReferential("TF").static.getNode_byLabel(f"{prefix}AJC")
        tf.static.addNode(f"{prefix}AJC",node_prox.m_global,positionType="Global",desc = node_prox.m_desc)

        # seg.addTrackingMarkerLabel("LAJC") # for LS fitting


    def _pelvis_Anatomicalcalibrate(self, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int) -> None:
        """Calibrate the anatomical referential of the pelvis segment.

        Constructs the anatomical referential for the pelvis segment based on static acquisition data and anatomical dictionary definitions.

        Args:
            aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
            dictAnatomic (Dict[str, Any]): Dictionary for reference markers used in anatomical coordinate system calibration.
            frameInit (int): The initial frame for calibration.
            frameEnd (int): The final frame for calibration.
        """


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

    def _thigh_Anatomicalcalibrate(self,side:str, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int) -> None:
        """Calibrate the anatomical referential of the thigh segment.

        Constructs the anatomical referential for the thigh segment based on static acquisition data and anatomical dictionary definitions.

        Args:
            side (str): body side
            aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
            dictAnatomic (Dict[str, Any]): Dictionary for reference markers used in anatomical coordinate system calibration.
            frameInit (int): The initial frame for calibration.
            frameEnd (int): The final frame for calibration.
        """
        prefix = "L" if side == "Left" else "R" if side == "Right" else ""

        seg=self.getSegment(f"{side} Thigh")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Thigh"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Thigh"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Thigh"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Thigh"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic[f"{side} Thigh"]['sequence'])

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
        hjc = seg.anatomicalFrame.static.getNode_byLabel(f"{prefix}HJC").m_local
        kjc = seg.anatomicalFrame.static.getNode_byLabel(f"{prefix}KJC").m_local

        seg.setLength(np.linalg.norm(kjc-hjc))


    # def _right_thigh_Anatomicalcalibrate(self, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int) -> None:
    #     """Calibrate the anatomical referential of the right thigh segment.

    #     Constructs the anatomical referential for the right thigh segment based on static acquisition data and anatomical dictionary definitions.

    #     Args:
    #         aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
    #         dictAnatomic (Dict[str, Any]): Dictionary for reference markers used in anatomical coordinate system calibration.
    #         frameInit (int): The initial frame for calibration.
    #         frameEnd (int): The final frame for calibration.
    #     """


    #     seg=self.getSegment("Right Thigh")

    #     # --- Construction of the anatomical Referential
    #     pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #     pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #     pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

    #     ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Thigh"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

    #     a1=(pt2-pt1)
    #     a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

    #     v=(pt3-pt1)
    #     v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

    #     a2=np.cross(a1,v)
    #     a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

    #     x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Right Thigh"]['sequence'])

    #     seg.anatomicalFrame.static.m_axisX=x
    #     seg.anatomicalFrame.static.m_axisY=y
    #     seg.anatomicalFrame.static.m_axisZ=z
    #     seg.anatomicalFrame.static.setRotation(R)
    #     seg.anatomicalFrame.static.setTranslation(ptOrigin)

    #     # --- relative rotation Technical Anatomical
    #     tf=seg.getReferential("TF")
    #     tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation()))

    #     # --- node manager
    #     for node in seg.getReferential("TF").static.getNodes():
    #         seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

    #     # --- compute lenght
    #     hjc = seg.anatomicalFrame.static.getNode_byLabel("RHJC").m_local
    #     kjc = seg.anatomicalFrame.static.getNode_byLabel("RKJC").m_local

    #     seg.setLength(np.linalg.norm(kjc-hjc))

    def _shank_Anatomicalcalibrate(self, side:str, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int) -> None:
        """Calibrate the anatomical referential of the shank segment.

        Constructs the anatomical referential for the shank segment based on static acquisition data and anatomical dictionary definitions.

        Args:
            side (str): body side
            aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
            dictAnatomic (Dict[str, Any]): Dictionary for reference markers used in anatomical coordinate system calibration.
            frameInit (int): The initial frame for calibration.
            frameEnd (int): The final frame for calibration.
        """

        prefix = "L" if side == "Left" else "R" if side == "Right" else ""

        seg=self.getSegment(f"{side} Shank")

        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Shank"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt2=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Shank"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
        pt3=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Shank"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Shank"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(pt3-pt1)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic[f"{side} Shank"]['sequence'])

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
        kjc = seg.anatomicalFrame.static.getNode_byLabel(f"{prefix}KJC").m_local
        ajc = seg.anatomicalFrame.static.getNode_byLabel(f"{prefix}AJC").m_local

        seg.setLength(np.linalg.norm(ajc-kjc))

    def _shankProximal_AnatomicalCalibrate(self,side:str, aquiStatic: btk.btkAcquisition, dictAnat: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrate the anatomical referential of the shank proximal segment.

        Constructs and adjusts the anatomical referential for the shank proximal segment based on static acquisition data, anatomical dictionary definitions, and optional parameters.

        Args:
            side (str): body side
            aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
            dictAnat (Dict[str, Any]): Dictionary for reference markers used in anatomical coordinate system calibration.
            frameInit (int): The initial frame for calibration.
            frameEnd (int): The final frame for calibration.
            options (Optional[Dict[str, Any]]): Additional options for calibration, if any.
        """
        prefix = "L" if side == "Left" else "R" if side == "Right" else ""

        tibialTorsion = 0.0
        if side == "Left":
            if self.m_useLeftTibialTorsion:
                tibialTorsion = -1.0*np.deg2rad(self.mp_computed["LeftTibialTorsionOffset"])
        elif side == "Right": 
            if self.m_useRightTibialTorsion:
                tibialTorsion = np.deg2rad(self.mp_computed["RightTibialTorsionOffset"])
  



        seg=self.getSegment(f"{side} Shank Proximal")


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


    def _foot_corrected_calibrate(self, side: str, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrate the anatomical referential of the corrected foot segment.

        Constructs the anatomical referential for the corrected foot segment based on static acquisition data, anatomical dictionary definitions, and optional parameters.

        Args:
            side (str): body side
            aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
            dictAnatomic (Dict[str, Any]): Dictionary for reference markers used in anatomical coordinate system calibration.
            frameInit (int): The initial frame for calibration.
            frameEnd (int): The final frame for calibration.
            options (Optional[Dict[str, Any]]): Additional options for calibration, if any.
        """

        prefix = "L" if side == "Left" else "R" if side == "Right" else ""

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


        seg=self.getSegment(f"{side} Foot")

        if "useBodyBuilderFoot" in options.keys() and options["useBodyBuilderFoot"]:
            LOGGER.logger.debug(f"You use a {side} corrected foot sequence different than native CGM1")
            dictAnatomic[f"{side} Foot"]={'sequence':"ZYX", 'labels':  [f"{prefix}TOE",f"{prefix}HEE",f"{prefix}KJC",f"{prefix}AJC"]}    # corrected foot


        # --- Construction of the anatomical Referential
        pt1=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Foot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # LTOE
        pt2=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Foot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0) # LHEE

        if side == "Left":
            if ("leftFlatFoot" in options.keys() and options["leftFlatFoot"]):
                LOGGER.logger.debug ("option (leftFlatFoot) enable")
                if ("LeftSoleDelta" in self.mp.keys() and self.mp["LeftSoleDelta"]!=0):
                    LOGGER.logger.debug ("option (LeftSoleDelta) compensation")

                pt2[2] = pt1[2]+self.mp['LeftSoleDelta']

        if side == "Right":
            if ("rightFlatFoot" in options.keys() and options["rightFlatFoot"]):
                LOGGER.logger.debug ("option (rightFlatFoot) enable")

                if ("RightSoleDelta" in self.mp.keys() and self.mp["RightSoleDelta"]!=0):
                    LOGGER.logger.debug ("option (RightSoleDelta) compensation")

                pt2[2] = pt1[2]+self.mp['RightSoleDelta']


        if dictAnatomic[f"{side} Foot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Foot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            v=(pt3-pt1)
        else:
            distalShank = self.getSegment(f"{side} Shank")
            proximalShank = self.getSegment(f"{side} Shank Proximal")
            if "viconCGM1compatible" in options.keys() and options["viconCGM1compatible"]:
                if distalShank.getReferential("TF").static.getNode_byLabel(f"{prefix}AJC").m_desc != "mid":
                    v=proximalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
                else:
                    v=distalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
            else:
                v=distalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)

        ptOrigin=aquiStatic.GetPoint(str(dictAnatomic[f"{side} Foot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

        a1=(pt2-pt1)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic[f"{side} Foot"]['sequence'])

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
        toe = seg.anatomicalFrame.static.getNode_byLabel(f"{prefix}TOE").m_local
        hee = seg.anatomicalFrame.static.getNode_byLabel(f"{prefix}HEE").m_local
        seg.setLength(np.linalg.norm(toe-hee)- markerDiameter/2.0)

        # com
        toe = seg.anatomicalFrame.static.getNode_byLabel(f"{prefix}TOE").m_global
        hee = seg.anatomicalFrame.static.getNode_byLabel(f"{prefix}HEE").m_global
        footLongAxis = (toe-hee)/np.linalg.norm(toe-hee)

        com = hee + 0.5 * seg.m_bsp["length"] * footLongAxis

        seg.anatomicalFrame.static.addNode("com",com,positionType="Global")


        # foot origin offset and Toe origin
        local_oo = np.array([-11, -11, -120])/169.0*seg.m_bsp["length"]
        local_to =local_oo + np.array([0, 0, -seg.m_bsp["length"]/3.0])

        seg.anatomicalFrame.static.addNode("FootOriginOffset",local_oo,positionType="Local")
        seg.anatomicalFrame.static.addNode("ToeOrigin",local_to,positionType="Local")

    # def _right_foot_corrected_calibrate(self, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
    #     """Calibrate the anatomical referential of the right corrected foot segment.

    #     Constructs the anatomical referential for the right corrected foot segment based on static acquisition data, anatomical dictionary definitions, and optional parameters.

    #     Args:
    #         aquiStatic (btk.btkAcquisition): Static acquisition data for calibration.
    #         dictAnatomic (Dict[str, Any]): Dictionary for reference markers used in anatomical coordinate system calibration.
    #         frameInit (int): The initial frame for calibration.
    #         frameEnd (int): The final frame for calibration.
    #         options (Optional[Dict[str, Any]]): Additional options for calibration, if any.
    #     """



    #     if "markerDiameter" in options.keys():
    #         LOGGER.logger.debug(" option (markerDiameter) found ")
    #         markerDiameter = options["markerDiameter"]
    #     else:
    #         markerDiameter=14.0

    #     if "basePlate" in options.keys():
    #         LOGGER.logger.debug(" option (basePlate) found ")
    #         basePlate = options["basePlate"]
    #     else:
    #         basePlate=2.0

    #     seg=self.getSegment("Right Foot")

    #     if "useBodyBuilderFoot" in options.keys() and options["useBodyBuilderFoot"]:
    #         LOGGER.logger.debug("You use a Right corrected foot sequence different than native CGM1")
    #         dictAnatomic["Right Foot"]={'sequence':"ZYX", 'labels':  ["RTOE","RHEE","RKJC","RAJC"]}    # corrected foot

    #     # --- Construction of the anatomical Referential
    #     pt1=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #     pt2=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #     #pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

    #     if ("rightFlatFoot" in options.keys() and options["rightFlatFoot"]):
    #         LOGGER.logger.debug ("option (rightFlatFoot) enable")

    #         if ("RightSoleDelta" in self.mp.keys() and self.mp["RightSoleDelta"]!=0):
    #             LOGGER.logger.debug ("option (RightSoleDelta) compensation")

    #         pt2[2] = pt1[2]+self.mp['RightSoleDelta']


    #     if dictAnatomic["Right Foot"]['labels'][2] is not None:
    #         pt3=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
    #         v=(pt3-pt1)
    #     else:
    #         distalShank = self.getSegment("Right Shank")
    #         proximalShank = self.getSegment("Right Shank Proximal")
    #         if "viconCGM1compatible" in options.keys() and options["viconCGM1compatible"]:
    #             if distalShank.getReferential("TF").static.getNode_byLabel("RAJC").m_desc != "mid":
    #                 v=proximalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
    #             else:
    #                 v=distalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)
    #         else:
    #             v=distalShank.anatomicalFrame.static.m_axisY #(pt3-pt1)


    #     ptOrigin=aquiStatic.GetPoint(str(dictAnatomic["Right Foot"]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

    #     a1=(pt2-pt1)
    #     a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

    #     v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

    #     a2=np.cross(a1,v)
    #     a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

    #     x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic["Right Foot"]['sequence'])

    #     seg.anatomicalFrame.static.m_axisX=x
    #     seg.anatomicalFrame.static.m_axisY=y
    #     seg.anatomicalFrame.static.m_axisZ=z
    #     seg.anatomicalFrame.static.setRotation(R)
    #     seg.anatomicalFrame.static.setTranslation(ptOrigin)

    #     # --- relative rotation Technical Anatomical
    #     tf=seg.getReferential("TF")
    #     # actual Relative Rotation
    #     trueRelativeMatrixAnatomic = np.dot(tf.static.getRotation().T,seg.anatomicalFrame.static.getRotation())
    #     y,x,z = euler.euler_yxz(trueRelativeMatrixAnatomic)

    #     # native CGM relative rotation
    #     rotX =np.array([[1,0,0],
    #                     [0,np.cos(x),-np.sin(x)],
    #                      [0,np.sin(x),np.cos(x)]])

    #     rotY =np.array([[np.cos(y),0,np.sin(y)],
    #                     [0,1,0],
    #                      [-np.sin(y),0,np.cos(y)]])

    #     relativeMatrixAnatomic = np.dot(rotY,rotX)

    #     tf.setRelativeMatrixAnatomic(relativeMatrixAnatomic)

    #     # --- node manager
    #     for node in seg.getReferential("TF").static.getNodes():
    #         seg.anatomicalFrame.static.addNode(node.getLabel(),node.getGlobal(),positionType="Global", desc = node.getDescription())

    #     # --- anthropo
    #     # length
    #     toe = seg.anatomicalFrame.static.getNode_byLabel("RTOE").m_local
    #     hee = seg.anatomicalFrame.static.getNode_byLabel("RHEE").m_local
    #     seg.setLength(np.linalg.norm(toe-hee)- markerDiameter/2.0)

    #     # com
    #     toe = seg.anatomicalFrame.static.getNode_byLabel("RTOE").m_global
    #     hee = seg.anatomicalFrame.static.getNode_byLabel("RHEE").m_global
    #     com = (toe+hee)/2.0

    #     seg.anatomicalFrame.static.addNode("com",com,positionType="Global")


    #     # foot origin offset and Toe origin
    #     local_oo = np.array([-11, 11, -120])/169.0*seg.m_bsp["length"]
    #     local_to =local_oo + np.array([0, 0, -seg.m_bsp["length"]/3.0])

    #     seg.anatomicalFrame.static.addNode("FootOriginOffset",local_oo,positionType="Local")
    #     seg.anatomicalFrame.static.addNode("ToeOrigin",local_to,positionType="Local")


    def _rotateAnatomicalFrame(self, segmentLabel: str, angle: float, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int) -> None:
        """Rotate the anatomical frame of a specified segment.

        Applies a rotation to the static anatomical referential of the specified segment.

        Args:
            segmentLabel (str): Label of the segment to be rotated.
            angle (float): Angle in degrees for rotation.
            aquiStatic (btk.btkAcquisition): Static acquisition data.
            dictAnatomic (Dict[str, Any]): Dictionary containing anatomical definitions.
            frameInit (int): Initial frame index.
            frameEnd (int): Ending frame index.
        """


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

    def getThighOffset(self, side: str = "both") -> None:
        """Calculate and store the thigh offset for specified side(s).

        Computes the angle between the projection of the lateral thigh marker and the knee flexion axis.

        Args:
            side (str, optional): Side of the body to compute the offset for. Can be 'both', 'left', or 'right'. Defaults to 'both'.
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



    def getShankOffsets(self, side: str = "both") -> None:
        """Calculate and store the shank offset for specified side(s).

        Computes the angle between the projection of the lateral shank marker and the ankle flexion axis.

        Args:
            side (str, optional): Side of the body to compute the offset for. Can be 'both', 'left', or 'right'. Defaults to 'both'.
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


    def getTibialTorsionOffset(self, side: str = "both") -> None:
        """Calculate and store the tibial torsion offset for specified side(s).

        Computes the tibial torsion offsets based on the anatomical frame of the thigh and shank segments.

        Args:
            side (str, optional): Side of the body to compute the offset for. Can be 'both', 'left', or 'right'. Defaults to 'both'.
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

    def getAbdAddAnkleJointOffset(self, side: str = "both") -> None:
        """Calculate and store the ankle abduction/adduction offset for specified side(s).

        Computes the angle in the frontal plane between the ankle marker and the ankle flexion axis.

        Args:
            side (str, optional): Side of the body to compute the offset for. Can be 'both', 'left', or 'right'. Defaults to 'both'.
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


    def getFootOffset(self, side: str = "both") -> None:
        """Calculate and store the foot offsets for specified side(s).

        Computes the plantar flexion offset and the rotation offset of the foot.

        Args:
            side (str, optional): Side of the body to compute the offset for. Can be 'both', 'left', or 'right'. Defaults to 'both'.
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

    # ---- Technical Referential Calibration
    def _head_calibrate(self, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrates the technical referential of the head segment.

        This method sets up the technical referential for the head segment using static acquisition data. It calculates the average position of specific markers within a given frame range and uses these positions to define the referential.

        Args:
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictRef (Dict[str, Any]): Technical referential definitions for the head.
            frameInit (int): The starting frame for averaging marker positions.
            frameEnd (int): The ending frame for averaging marker positions.
            options (Optional[Dict[str, Any]]): Additional options for calibration, like marker diameter and base plate size.
        """


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

    def _head_AnatomicalCalibrate(self, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrates the anatomical referential of the head segment.

        This method sets up the anatomical referential for the head segment using static acquisition data. It calculates the average position of specific anatomical landmarks within a given frame range and uses these to define the referential.

        Args:
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictAnatomic (Dict[str, Any]): Anatomical referential definitions for the head.
            frameInit (int): The starting frame for averaging landmark positions.
            frameEnd (int): The ending frame for averaging landmark positions.
            options (Optional[Dict[str, Any]]): Additional options for calibration (not used in this method).
        """


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



    def _torso_calibrate(self, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrates the technical referential of the torso (thorax) segment.

        This method establishes the technical referential for the torso segment using static acquisition data. It computes the mean position of specified markers over a range of frames and utilizes these positions to construct the referential.

        Args:
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictRef (Dict[str, Any]): Technical referential definitions for the torso.
            frameInit (int): The initial frame for averaging marker positions.
            frameEnd (int): The final frame for averaging marker positions.
            options (Optional[Dict[str, Any]]): Additional options for calibration, such as marker diameter and base plate size.
        """


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



    def _torso_Anatomicalcalibrate(self, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int) -> None:
        """Calibrates the anatomical referential of the torso (thorax) segment.

        This method sets up the anatomical referential for the torso segment based on static acquisition data. It calculates the average position of specific anatomical landmarks within a given frame range to define the referential.

        Args:
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictAnatomic (Dict[str, Any]): Anatomical referential definitions for the torso.
            frameInit (int): The starting frame for averaging landmark positions.
            frameEnd (int): The ending frame for averaging landmark positions.
        """


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


    def _clavicle_calibrate(self, side: str, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrates the technical referential of the clavicle segment for a specified side.

        This function constructs the technical referential for either the or right clavicle segment using static motion data. It computes the mean position of markers within a specified frame range for this purpose.

        Args:
            side (str): Specifies the side ('Left' or 'Right') of the clavicle.
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictR
        """

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


    def _clavicle_Anatomicalcalibrate(self, side: str, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int) -> None:
        """Calibrates the anatomical referential of the clavicle segment for a specified side.

        This method establishes the anatomical referential for either the or right clavicle segment based on static acquisition data. It calculates the average position of anatomical landmarks within a frame range for this purpose.

        Args:
            side (str): Specifies the side ('Left' or 'Right') of the clavicle.
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictAnatomic (Dict[str, Any]): Anatomical referential definitions for the clavicle.
            frameInit (int): The starting frame for averaging landmark positions.
            frameEnd (int): The ending frame for averaging landmark positions.
        """



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


    def _constructArmVirtualMarkers(self, side: str, aqui: btk.btkAcquisition) -> None:
        """Constructs virtual markers for the arm on the specified side.

        This method calculates virtual markers such as the mid-wrist point and the virtual wand, which are essential for biomechanical analysis of arm movements.

        Args:
            side (str): The side of the body ('Left' or 'Right').
            aqui (btk.btkAcquisition): The motion capture data.
        """


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


    def _upperArm_calibrate(self, side: str, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrates the upper arm segment for the specified side.

        This method sets up the technical referential for the upper arm segment using static acquisition data. It calculates the mean position of specified markers over a range of frames for this purpose.

        Args:
            side (str): The side of the body ('Left' or 'Right').
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictRef (Dict[str, Any]): Definitions for the technical referential of the upper arm.
            frameInit (int): The initial frame for averaging marker positions.
            frameEnd (int): The final frame for averaging marker positions.
            options (Optional[Dict[str, Any]]): Additional calibration options.
        """


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


    def _upperArm_Anatomicalcalibrate(self, side: str, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int) -> None:
        """Calibrates the anatomical referential of the upper arm segment for a specified side.

        This method establishes the anatomical referential for the upper arm segment based on static acquisition data. It utilizes the average position of specific anatomical landmarks within a given frame range.

        Args:
            side (str): The side of the body ('Left' or 'Right').
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictAnatomic (Dict[str, Any]): Anatomical referential definitions for the upper arm.
            frameInit (int): The starting frame for averaging landmark positions.
            frameEnd (int): The ending frame for averaging landmark positions.
        """


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

    def _foreArm_calibrate(self, side: str, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrates the technical referential of the forearm segment for the specified side.

        This function constructs the technical referential for either the or right forearm segment using static motion data. It calculates the mean position of markers within a specified frame range.

        Args:
            side (str): The side of the body ('Left' or 'Right').
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictRef (Dict[str, Any]): Technical referential definitions for the forearm.
            frameInit (int): The initial frame for averaging marker positions.
            frameEnd (int): The final frame for averaging marker positions.
            options (Optional[Dict[str, Any]]): Additional calibration options.
        """


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

    def _foreArm_Anatomicalcalibrate(self, side: str, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int) -> None:
        """Calibrates the anatomical referential of the forearm segment for a specified side.

        This method sets up the anatomical referential for the forearm segment based on static acquisition data. It utilizes the average position of anatomical landmarks within a given frame range.

        Args:
            side (str): The side of the body ('Left' or 'Right').
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictAnatomic (Dict[str, Any]): Anatomical referential definitions for the forearm.
            frameInit (int): The starting frame for averaging landmark positions.
            frameEnd (int): The ending frame for averaging landmark positions.
        """



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


    def _hand_calibrate(self, side: str, aquiStatic: btk.btkAcquisition, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None) -> None:
        """Calibrates the technical referential of the hand segment for the specified side.

        This function establishes the technical referential for either the or right hand segment using static motion data. It calculates the mean position of markers within a specified frame range.

        Args:
            side (str): The side of the body ('Left' or 'Right').
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictRef (Dict[str, Any]): Technical referential definitions for the hand.
            frameInit (int): The initial frame for averaging marker positions.
            frameEnd (int): The final frame for averaging marker positions.
            options (Optional[Dict[str, Any]]): Additional calibration options.
        """


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


    def _hand_Anatomicalcalibrate(self, side: str, aquiStatic: btk.btkAcquisition, dictAnatomic: Dict[str, Any], frameInit: int, frameEnd: int) -> None:
        """Calibrates the anatomical referential of the hand segment for a specified side.

        This method establishes the anatomical referential for the hand segment based on static acquisition data. It calculates the average position of anatomical landmarks within a given frame range.

        Args:
            side (str): The side of the body ('Left' or 'Right').
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictAnatomic (Dict[str, Any]): Anatomical referential definitions for the hand.
            frameInit (int): The starting frame for averaging landmark positions.
            frameEnd (int): The ending frame for averaging landmark positions.
        """


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

    # ----- Motion --------------
    def computeOptimizedSegmentMotion(self, aqui: btk.btkAcquisition, segments: List[str], dictRef: Dict[str, Any], dictAnat: Dict[str, Any], motionMethod: enums.motionMethod, options: Dict[str, Any]) -> None:
        """Compute poses of both Technical and Anatomical coordinate systems for specific segments of the model.

        Args:
            aqui (btk.btkAcquisition): Motion acquisition data.
            segments (List[str]): Segments of the model.
            dictRef (Dict[str, Any]): Technical referential definitions.
            dictAnat (Dict[str, Any]): Anatomical referential definitions.
            motionMethod (enums.motionMethod): Segmental motion method to apply.
            options (Dict[str, Any]): Passed known-arguments.
        """



        # ---remove all  direction marker from tracking markers.
        if self.staExpert:
            for seg in self.m_segmentCollection:
                selectedTrackingMarkers=[]
                for marker in seg.m_tracking_markers:
                    if marker in self.__class__.TRACKING_MARKERS : # get class variable MARKER even from child
                        selectedTrackingMarkers.append(marker)
                seg.m_tracking_markers= selectedTrackingMarkers


        LOGGER.logger.debug("--- Segmental Least-square motion process ---")
        if "Pelvis" in segments:
            self._pelvis_motion_optimize(aqui, dictRef, motionMethod)
            self._anatomical_motion(aqui,"Pelvis",originLabel = str(dictAnat["Pelvis"]['labels'][3]))

        if "Left Thigh" in segments:
            self._thigh_motion_optimize("Left",aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Thigh",originLabel = "LKJC")


        if "Right Thigh" in segments:
            self._thigh_motion_optimize("Right",aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Thigh",originLabel = "RKJC")


        if "Left Shank" in segments:
            self._shank_motion_optimize("Left",aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Shank",originLabel = "LAJC")

        if "Right Shank" in segments:
            self._shank_motion_optimize("Right",aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Shank",originLabel = "RAJC")

        if "Left Foot" in segments:
            self._foot_motion_optimize("Left",aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Left Foot",originLabel = "LHEE")

        if "Right Foot" in segments:
            self._foot_motion_optimize("Right",aqui, dictRef,motionMethod)
            self._anatomical_motion(aqui,"Right Foot",originLabel = "RHEE")


    

    def computeMotion(self, aqui: btk.btkAcquisition, dictRef: Dict[str, Any], dictAnat: Dict[str, Any], motionMethod: enums.motionMethod, options: Optional[Dict[str, Any]] = None) -> None:
        """Compute poses of both Technical and Anatomical coordinate systems.

        Args:
            aqui (btk.btkAcquisition): Acquisition data.
            dictRef (Dict[str, Any]): Technical referential definitions.
            dictAnat (Dict[str, Any]): Anatomical referential definitions.
            motionMethod (enums.motionMethod): Segmental motion method.
            options (Optional[Dict[str, Any]], optional): Passed arguments to embedded functions. Defaults to None.

        Options:
            pigStatic (bool): Compute foot coordinate system according to the Vicon Plugin-gait.
            forceFoot6DoF (bool): Apply 6DOF pose optimization on the foot.
        """

        

        LOGGER.logger.debug("=====================================================")
        LOGGER.logger.debug("===================  CGM MOTION   ===================")
        LOGGER.logger.debug("=====================================================")

        pigStaticProcessing= True if "pigStatic" in options.keys() and options["pigStatic"] else False
        forceFoot6DoF= True if "forceFoot6DoF" in options.keys() and options["forceFoot6DoF"] else False


        if motionMethod == enums.motionMethod.Determinist: #cmf.motionMethod.Native:

            self._pelvis_motion(aqui, dictRef, dictAnat)
            self._thorax_motion(aqui, dictRef,dictAnat,options=options)
            self._head_motion(aqui, dictRef,dictAnat,options=options)

            self._processLowerMotion("Left",pigStaticProcessing,aqui,dictRef,dictAnat,options)
            self._processLowerMotion( "Right",pigStaticProcessing,aqui,dictRef,dictAnat,options)
            self._processUpperMotion("Left",aqui,dictRef,dictAnat,options)
            self._processUpperMotion("Right",aqui,dictRef,dictAnat,options)
            
            

 

        if motionMethod == enums.motionMethod.Sodervisk:

            # ---remove all  direction marker from tracking markers.
            if self.staExpert:
                for seg in self.m_segmentCollection:
                    selectedTrackingMarkers=[]
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

            self._thorax_motion(aqui, dictRef,dictAnat,options=options)
            self._head_motion(aqui, dictRef,dictAnat,options=options)

            start = time.time()
            self._processLowerMotionOptimize("Left",forceFoot6DoF,aqui,dictRef,dictAnat,options,motionMethod)
            self._processLowerMotionOptimize("Right",forceFoot6DoF,aqui,dictRef,dictAnat,options,motionMethod)
            self._processUpperMotion("Left",aqui,dictRef,dictAnat,options)
            self._processUpperMotion("Right",aqui,dictRef,dictAnat,options)
            end = time.time()
            # total time taken
            print(f"Runtime of the program is {end - start}")
   


    def _processLowerMotion(self,side,pigStaticProcessing,aqui,dictRef,dictAnat,options):
        self._thigh_motion(side,aqui, dictRef, dictAnat,options=options)

        # if rotation offset from knee functional calibration methods
        if self.mp_computed[f"{side}KneeFuncCalibrationOffset"]:
            offset = self.mp_computed[f"{side}KneeFuncCalibrationOffset"]
            self._rotate_anatomical_motion(f"{side} Thigh",offset,
                                    aqui,options=options)

        self._shank_motion(side,aqui, dictRef, dictAnat,options=options)
        self._shankProximal_motion(side,aqui,dictAnat,options=options)

        if pigStaticProcessing:
            self._foot_motion_static(side,aqui, dictAnat,options=options)
        else:
            self._foot_motion(side,aqui, dictRef, dictAnat,options=options)     
    

    def _processUpperMotion(self,side,aqui,dictRef,dictAnat,options):
        self._clavicle_motion(side,aqui, dictRef,dictAnat,options=options)
        self._constructArmVirtualMarkers(side, aqui)
        self._upperArm_motion(side,aqui, dictRef,dictAnat,options=options,   frameReconstruction="Technical")
        self._foreArm_motion(side,aqui, dictRef,dictAnat,options=options, frameReconstruction="Technical")
        self._upperArm_motion(side,aqui, dictRef,dictAnat,options=options,   frameReconstruction="Anatomical")
        self._foreArm_motion(side,aqui, dictRef,dictAnat,options=options, frameReconstruction="Anatomical")
        self._hand_motion(side,aqui, dictRef,dictAnat,options=options)


    def _processLowerMotionOptimize(self,side,forceFoot6DoF,aqui,dictRef,dictAnat,options,motionMethod):
        self._thigh_motion_optimize(side,aqui, dictRef,motionMethod)
        self._anatomical_motion(aqui,f"{side} Thigh",originLabel = str(dictAnat[f"{side} Thigh"]['labels'][3]))

        self._shank_motion_optimize(side,aqui, dictRef,motionMethod)
        self._anatomical_motion(aqui,f"{side} Shank",originLabel = str(dictAnat[f"{side} Shank"]['labels'][3]))
        self._shankProximal_motion(side,aqui,dictAnat,options=options)

        if forceFoot6DoF:
            # foot
            # issue with least-square optimization :  AJC - HEE and TOE may be inline -> singularities !!
            self._foot_motion_optimize(side,aqui, dictRef,dictAnat, motionMethod)
            self._anatomical_motion(aqui,f"{side} Foot",originLabel = str(dictAnat[f"{side} Foot"]['labels'][3]))
        else:
            self._foot_motion(side,aqui, dictRef, dictAnat,options=options)



    
    def _pelvis_motion(self, aqui: btk.btkAcquisition, dictRef: Dict[str, Any], dictAnat: Dict[str, Any]) -> None:
        """Process the motion of the pelvis segment.

        Args:
            aqui (btk.btkAcquisition): Motion acquisition data.
            dictRef (Dict[str, Any]): Technical referential definitions for the pelvis.
            dictAnat (Dict[str, Any]): Anatomical referential definitions for the pelvis.
        """


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

        pt1=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][0])).GetValues()
        pt2=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][1])).GetValues()
        pt3=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][2])).GetValues()
        ptOrigin=aqui.GetPoint(str(dictRef["Pelvis"]["TF"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Pelvis"]["TF"]['sequence'])

        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.getReferential("TF").addMotionFrame(csFrame)

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

        lhjc = aqui.GetPoint("LHJC").GetValues()
        rhjc =  aqui.GetPoint("RHJC").GetValues()

        # additional markers
        val=(aqui.GetPoint("LHJC").GetValues() + aqui.GetPoint("RHJC").GetValues()) / 2.0
        btkTools.smartAppendPoint(aqui,"midHJC",val,desc="")

        pt1=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][0])).GetValues()
        pt2=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][1])).GetValues()
        pt3=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][2])).GetValues()
        ptOrigin=aqui.GetPoint(str(dictAnat["Pelvis"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
        x,y,z,R=frame.setFrameData(a1,a2,dictAnat["Pelvis"]['sequence'])

        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.anatomicalFrame.addMotionFrame(csFrame)

            # length
            pelvisScale = np.linalg.norm(lhjc[i,:]-rhjc[i,:])
            offset = (lhjc[i,:]+rhjc[i,:])/2.0

            TopLumbar5[i,:] = offset +  np.dot(R[i],(np.array([ 0, 0, 0.925]))* pelvisScale)


        self._TopLumbar5 = TopLumbar5



    
    def _thigh_motion(self,side:str, aqui: btk.btkAcquisition, dictRef: Dict[str, Any], dictAnat: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        """Process the motion of the thigh segment.

        Args:
            side (str): body side (Left or Right)
            aqui (btk.btkAcquisition): Motion acquisition data.
            dictRef (Dict[str, Any]): Technical referential definitions for the thigh.
            dictAnat (Dict[str, Any]): Anatomical referential definitions for the thigh.
            options (Optional[Dict[str, Any]], optional): Additional options. Defaults to None.
        """
        prefix = "L" if side == "Left" else "R" if side == "Right" else ""
        sign = -1 if side == "Left" else 1 if side == "Right" else 0

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

        seg=self.getSegment(f"{side} Thigh")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]   # reinit Technical Frame Motion ()

        # additional markers
        # NA

        # computation
                # --- LKJC
        KJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))
        

        pt1=aqui.GetPoint(str(dictRef[f"{side} Thigh"]["TF"]['labels'][0])).GetValues()
        pt2=aqui.GetPoint(str(dictRef[f"{side} Thigh"]["TF"]['labels'][1])).GetValues()
        pt3=aqui.GetPoint(str(dictRef[f"{side} Thigh"]["TF"]['labels'][2])).GetValues()
        ptOrigin=aqui.GetPoint(str(dictRef[f"{side} Thigh"]["TF"]['labels'][3])).GetValues()
        
        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
        x,y,z,R=frame.setFrameData(a1,a2,dictRef[f"{side} Thigh"]["TF"]['sequence'])


        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.getReferential("TF").addMotionFrame(csFrame)

            if validFrames[i]:
                KJCvalues[i,:] = modelDecorator.VCMJointCentre( (self.mp[f"{side}KneeWidth"]+ markerDiameter)/2.0 ,pt1[i,:],pt2[i,:],pt3[i,:], beta=sign*self.mp_computed[f"{side}ThighRotationOffset"] )

        if  f"use{side}KJCmarker" in options.keys() and options[f"use{side}KJCmarker"] != f"{prefix}KJC":
            LOGGER.logger.info(f"[pyCGM2] - {prefix}KJC marker forced to use %s"%(options[f"use{side}KJCmarker"]))
            KJCvalues = aqui.GetPoint(options[f"use{side}KJCmarker"]).GetValues()
            desc = aqui.GetPoint(options[f"use{side}KJCmarker"]).GetDescription()
            btkTools.smartAppendPoint(aqui,f"{prefix}KJC",KJCvalues,desc=str(desc))
        else:
            desc = seg.getReferential('TF').static.getNode_byLabel(f"{prefix}KJC").m_desc
            btkTools.smartAppendPoint(aqui,f"{prefix}KJC",KJCvalues,desc=str("Chord-"+desc))

        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA
        # computation
        pt1=aqui.GetPoint(str(dictAnat[f"{side} Thigh"]['labels'][0])).GetValues()
        pt2=aqui.GetPoint(str(dictAnat[f"{side} Thigh"]['labels'][1])).GetValues()
        pt3=aqui.GetPoint(str(dictAnat[f"{side} Thigh"]['labels'][2])).GetValues()
        ptOrigin=aqui.GetPoint(str(dictAnat[f"{side} Thigh"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
        x,y,z,R=frame.setFrameData(a1,a2,dictAnat[f"{side} Thigh"]['sequence'])

        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.anatomicalFrame.addMotionFrame(csFrame)

    

    
    def _shank_motion(self,side:str, aqui: btk.btkAcquisition, dictRef: Dict[str, Any], dictAnat: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        """Process the motion of the shank segment.

        Args:
            side (str): body side (Left or Right)
            aqui (btk.btkAcquisition): Motion acquisition data.
            dictRef (Dict[str, Any]): Technical referential definitions for the shank.
            dictAnat (Dict[str, Any]): Anatomical referential definitions for the shank.
            options (Optional[Dict[str, Any]], optional): Additional options. Defaults to None.
        """
        prefix = "L" if side == "Left" else "R" if side == "Right" else ""
        sign = -1 if side == "Left" else 1 if side == "Right" else 0

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

        seg=self.getSegment(f"{side} Shank")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA


        # --- LAJC
        # computation
        AJCvalues=np.zeros((aqui.GetPointFrameNumber(),3))

        pt1=aqui.GetPoint(str(dictRef[f"{side} Shank"]["TF"]['labels'][0])).GetValues() #ANK
        pt2=aqui.GetPoint(str(dictRef[f"{side} Shank"]["TF"]['labels'][1])).GetValues() #KJC
        pt3=aqui.GetPoint(str(dictRef[f"{side} Shank"]["TF"]['labels'][2])).GetValues() #TIB
        ptOrigin=aqui.GetPoint(str(dictRef[f"{side} Shank"]["TF"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
        x,y,z,R=frame.setFrameData(a1,a2,dictRef[f"{side} Shank"]["TF"]['sequence'])

        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.getReferential("TF").addMotionFrame(csFrame)

            if validFrames[i]:
                AJCvalues[i,:] = modelDecorator.VCMJointCentre( (self.mp[f"{side}AnkleWidth"]+ markerDiameter)/2.0 ,pt1[i,:],pt2[i,:],pt3[i,:], beta=sign*self.mp_computed[f"{side}ShankRotationOffset"] )
                # update of the AJC location with rotation around abdAddAxis
                AJCvalues[i,:] = self._rotateAjc(AJCvalues[i,:],pt2[i,:],pt1[i,:],self.mp_computed[f"{side}AnkleAbAddOffset"])


        if  f"use{side}AJCmarker" in options.keys() and options[f"use{side}AJCmarker"] != f"{prefix}AJC":
            LOGGER.logger.info(f"[pyCGM2] - {prefix}AJC marker forced to use %s"%(options[f"use{side}AJCmarker"]))
            AJCvalues = aqui.GetPoint(options[f"use{side}AJCmarker"]).GetValues()
            desc = aqui.GetPoint(options[f"use{side}AJCmarker"]).GetDescription()
            btkTools.smartAppendPoint(aqui,f"{prefix}AJC",AJCvalues,desc=desc)
        else:
            # --- LAJC
            desc_node = seg.getReferential('TF').static.getNode_byLabel(f"{prefix}AJC").m_desc
            if self.mp_computed[f"{side}AnkleAbAddOffset"] > 0.01:
                desc="chord+AbAdRot-"+desc_node
            else:
                desc="chord "+desc_node
            btkTools.smartAppendPoint(aqui,f"{prefix}AJC",AJCvalues,desc=desc)


        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA
        pt1=aqui.GetPoint(str(dictAnat[f"{side} Shank"]['labels'][0])).GetValues()
        pt2=aqui.GetPoint(str(dictAnat[f"{side} Shank"]['labels'][1])).GetValues()
        pt3=aqui.GetPoint(str(dictAnat[f"{side} Shank"]['labels'][2])).GetValues()
        ptOrigin=aqui.GetPoint(str(dictAnat[f"{side} Shank"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
        x,y,z,R=frame.setFrameData(a1,a2,dictAnat[f"{side} Shank"]['sequence'])
        # computation
        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.anatomicalFrame.addMotionFrame(csFrame)

    
    def _shankProximal_motion(self, side:str, aqui: btk.btkAcquisition, dictAnat: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        """Process the motion of the shank proximal segment.

        Args:
            side (str): body side (Left or Right)
            aqui (btk.btkAcquisition): Motion acquisition data.
            dictAnat (Dict[str, Any]): Anatomical referential definitions for the shank proximal.
            options (Optional[Dict[str, Any]], optional): Additional options. Defaults to None.
        """
        prefix = "L" if side == "Left" else "R" if side == "Right" else ""
        sign = -1 if side == "Left" else 1 if side == "Right" else 0

        seg=self.getSegment(f"{side} Shank")
        segProx=self.getSegment(f"{side} Shank Proximal")


        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        segProx.setExistFrames(validFrames)


        # --- managment of tibial torsion
        if side == "Left":
            if self.m_useLeftTibialTorsion:
                tibialTorsion = sign*np.deg2rad(self.mp_computed[f"{side}TibialTorsionOffset"])
            else:
                tibialTorsion = 0.0

        if side == "Right":
            if self.m_useRightTibialTorsion:
                tibialTorsion = sign*np.deg2rad(self.mp_computed[f"{side}TibialTorsionOffset"])
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
        KJC = aqui.GetPoint(str(dictAnat[f"{side} Shank"]['labels'][3]))

        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            ptOrigin=KJC.GetValues()[i,:]
            segProx.getReferential("TF").addMotionFrame(seg.getReferential("TF").motion[i]) # copy technical shank

            R = np.dot(seg.anatomicalFrame.motion[i].getRotation(),rotZ_tibRot) # affect Tibial torsion to anatomical shank

            csFrame.update(R,ptOrigin)
            segProx.anatomicalFrame.addMotionFrame(csFrame)


    
    def _foot_motion(self, side:str, aqui: btk.btkAcquisition, dictRef: Dict[str, Any], dictAnat: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        """Process the motion of the foot segment.

        Args:
            side (str): body side (Left or Right)
            aqui (btk.btkAcquisition): Motion acquisition data.
            dictRef (Dict[str, Any]): Technical referential definitions for the foot.
            dictAnat (Dict[str, Any]): Anatomical referential definitions for the foot.
            options (Optional[Dict[str, Any]], optional): Additional options. Defaults to None.
        """

        prefix = "L" if side == "Left" else "R" if side == "Right" else ""
        sign = -1 if side == "Left" else 1 if side == "Right" else 0

        seg=self.getSegment(f"{side} Foot")
        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        # --- motion of the technical referential
        seg.getReferential("TF").motion =[]

        # additional markers
        # NA

        # computation
        pt1=aqui.GetPoint(str(dictRef[f"{side} Foot"]["TF"]['labels'][0])).GetValues() #toe
        pt2=aqui.GetPoint(str(dictRef[f"{side} Foot"]["TF"]['labels'][1])).GetValues() #ajc

        if dictRef[f"{side} Foot"]["TF"]['labels'][2] is not None:
            pt3=aqui.GetPoint(str(dictRef[f"{side} Foot"]["TF"]['labels'][2])).GetValues()
            v=(pt3-pt1)
        else:
            pt3 = None
            v=np.vstack([obj.m_axisY for obj in self.getSegment(f"{side} Shank Proximal").anatomicalFrame.motion])
                

        ptOrigin=aqui.GetPoint(str(dictRef[f"{side} Foot"]["TF"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3,v=v)
        x,y,z,R=frame.setFrameData(a1,a2,dictRef[f"{side} Foot"]["TF"]['sequence'])


        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            if "viconCGM1compatible" in options.keys() and options["viconCGM1compatible"]:
                R2 = R[i]
            else:
                R2 = np.dot(R[i],self._R_leftUnCorrfoot_dist_prox) if side == "Left" else np.dot(R[i],self._R_rightUnCorrfoot_dist_prox)


            csFrame.m_axisX=R2[:,0]
            csFrame.m_axisY=R2[:,1]
            csFrame.m_axisZ=R2[:,2]
            csFrame.setRotation(R2)
            csFrame.setTranslation(ptOrigin[i])

            seg.getReferential("TF").addMotionFrame(csFrame)


        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA

        # computation
        ptOrigin=aqui.GetPoint(str(dictAnat[f"{side} Foot"]['labels'][3])).GetValues()
        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            csFrame.update(R,ptOrigin[i,:])
            seg.anatomicalFrame.addMotionFrame(csFrame)


    
    def _foot_motion_static(self,side:str, aquiStatic: btk.btkAcquisition, dictAnat: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        """Process the static motion of the foot segment.

        This method is used for static motion analysis of the foot, taking into account options like flat foot condition.

        Args:
            side (str): body side (Left or Right)
            aquiStatic (btk.btkAcquisition): Static motion acquisition data.
            dictAnat (Dict[str, Any]): Anatomical referential definitions for the foot.
            options (Optional[Dict[str, Any]], optional): Additional options including flat foot settings. Defaults to None.
        """

        prefix = "L" if side == "Left" else "R" if side == "Right" else ""
        sign = -1 if side == "Left" else 1 if side == "Right" else 0

        seg=self.getSegment(f"{side} Foot")

        validFrames = btkTools.getValidFrames(aquiStatic,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA


        ptOrigin=aquiStatic.GetPoint(str(dictAnat[f"{side} Foot"]['labels'][3])).GetValues()
        pt1=aquiStatic.GetPoint(str(dictAnat[f"{side} Foot"]['labels'][0])).GetValues() #toe
        pt2=aquiStatic.GetPoint(str(dictAnat[f"{side} Foot"]['labels'][1])).GetValues() #hee


        if side == "Left":
            if ("leftFlatFoot" in options.keys() and options["leftFlatFoot"]):
                pt2[:,2] = pt1[:,2]+self.mp["LeftSoleDelta"]
        if side == "Right":
            if ("rightFlatFoot" in options.keys() and options["rightFlatFoot"]):
                pt2[:,2] = pt1[:,2]+self.mp['RightSoleDelta']

        if dictAnat[f"{side} Foot"]['labels'][2] is not None:
            pt3=aquiStatic.GetPoint(str(dictAnat[f"{side} Foot"]['labels'][2])).GetValues()
            v=(pt3-pt1)
        else:
            pt3 = None
            v=np.vstack([obj.m_axisY for obj in self.getSegment(f"{side} Shank").anatomicalFrame.motion])

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3,v=v)
        x,y,z,R=frame.setFrameData(a1,a2,dictAnat[f"{side} Foot"]['sequence'])

        # computation
        for i in range(0,aquiStatic.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.update(R[i],ptOrigin[i,:])
            seg.anatomicalFrame.addMotionFrame(csFrame)


    
    # ----- least-square Segmental motion ------
    
    def _pelvis_motion_optimize(self, aqui: btk.btkAcquisition, dictRef: Dict[str, Any], motionMethod: enums.motionMethod,
                                 anatomicalFrameMotionEnable: bool = True) -> None:
        """Optimize the motion of the pelvis segment.

        Args:
            aqui (btk.btkAcquisition): Motion acquisition data.
            dictRef (Dict[str, Any]): Technical referential definitions for the pelvis.
            motionMethod (enums.motionMethod): Method for motion optimization.
            anatomicalFrameMotionEnable (bool, optional): Enable motion for anatomical frame. Defaults to True.
        """


        seg=self.getSegment("Pelvis")
        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        validFrames = np.array(validFrames)
        invalid_indices = ~validFrames

        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

        # part 1: get global location in Static
        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.array([seg.getReferential("TF").static.getNode_byLabel(label).m_global 
                                        for label in seg.m_tracking_markers])


        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        point_values = {label: aqui.GetPoint(label).GetValues() for label in seg.m_tracking_markers}
        n_frames = aqui.GetPointFrameNumber()

        for i in range(0,n_frames):
            csFrame=frame.Frame()
            dynPos = np.array([point_values[label][i, :] for label in seg.m_tracking_markers])

            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(staticPos,dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt

                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)

            seg.getReferential("TF").addMotionFrame(csFrame)


        # --- HJC
        values_LHJCnode=seg.getReferential('TF').getNodeTrajectory("LHJC")
        values_RHJCnode=seg.getReferential('TF').getNodeTrajectory("RHJC")
        
        values_LHJCnode[invalid_indices] = np.zeros(3)
        values_RHJCnode[invalid_indices] = np.zeros(3)

        # import ipdb; ipdb.set_trace()
        # for i in range(0,aqui.GetPointFrameNumber()):
        #     if not validFrames[i]:
        #         values_LHJCnode[i,:] = np.zeros(3)
        #         values_RHJCnode[i,:] = np.zeros(3)


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
    
    

    
    def _thigh_motion_optimize(self,side:str, aqui: btk.btkAcquisition, dictRef: Dict[str, Any], motionMethod: enums.motionMethod) -> None:
        """Optimizes the motion of the thigh segment using a specific motion method.

        This method adjusts the technical frame of the thigh segment based on dynamic position data. It optionally adds LHJC to the tracking markers if needed.

        Args:
            aqui (btk.btkAcquisition): Motion acquisition data.
            side (str): body side ( Left or Right)
            dictRef (Dict[str, Any]): Technical referential definitions for the thigh.
            motionMethod (enums.motionMethod): Segmental motion method to apply.
        """
        
        prefix = "L" if side == "Left" else "R" if side == "Right" else ""

        seg=self.getSegment(side+" Thigh")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)

        validFrames = np.array(validFrames)
        invalid_indices = ~validFrames

        #  --- add LHJC if list <2 - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if f"{prefix}HJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append(f"{prefix}HJC")
                    LOGGER.logger.debug(f"{prefix}HJC added to tracking marker list")



        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

        

        # part 1: get global location in Static
        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.array([seg.getReferential("TF").static.getNode_byLabel(label).m_global 
                                        for label in seg.m_tracking_markers])


        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        point_values = {label: aqui.GetPoint(label).GetValues() for label in seg.m_tracking_markers}
        n_frames = aqui.GetPointFrameNumber()

        for i in range(0,n_frames):
            csFrame=frame.Frame()
            dynPos = np.array([point_values[label][i, :] for label in seg.m_tracking_markers])

            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(staticPos,dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt
                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)

            seg.getReferential("TF").addMotionFrame(csFrame)

        # --- LKJC
        desc = seg.getReferential('TF').static.getNode_byLabel(f"{prefix}KJC").m_desc
        values_KJCnode=seg.getReferential('TF').getNodeTrajectory(f"{prefix}KJC")

        values_KJCnode[invalid_indices] = np.zeros(3)

        btkTools.smartAppendPoint(aqui,f"{prefix}KJC",values_KJCnode, desc=str("opt-"+desc))


        # --- LHJC from Thigh
        # values_HJCnode=seg.getReferential('TF').getNodeTrajectory("LHJC")
        # btkTools.smartAppendPoint(aqui,"LHJC-Thigh",values_HJCnode, desc="opt from Thigh")

        # remove LHC from list of tracking markers
        if f"{prefix}LHJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove(f"{prefix}HJC")




    
    
    def _shank_motion_optimize(self,side:str, aqui: btk.btkAcquisition, dictRef: Dict[str, Any], motionMethod: enums.motionMethod) -> None:
        """Optimizes the motion of the shank segment using a specific motion method.

        This method adjusts the technical frame of the shank segment based on dynamic position data. It optionally adds LKJC to the tracking markers if needed.

        Args:
            aqui (btk.btkAcquisition): Motion acquisition data.
            side (str): body side ( Left or Right)
            dictRef (Dict[str, Any]): Technical referential definitions for the shank.
            motionMethod (enums.motionMethod): Segmental motion method to apply.
        """
        prefix = "L" if side == "Left" else "R" if side == "Right" else ""

        seg=self.getSegment(f"{side} Shank")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)
        validFrames = np.array(validFrames)
        invalid_indices = ~validFrames

        #  --- add LKJC if list <2 - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if f"{prefix}KJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append(f"{prefix}KJC")
                    LOGGER.logger.debug(f"{prefix}KJC added to tracking marker list")

        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

       # part 1: get global location in Static
        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.array([seg.getReferential("TF").static.getNode_byLabel(label).m_global 
                                        for label in seg.m_tracking_markers])


        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        point_values = {label: aqui.GetPoint(label).GetValues() for label in seg.m_tracking_markers}
        n_frames = aqui.GetPointFrameNumber()

        for i in range(0,n_frames):
            csFrame=frame.Frame()
            dynPos = np.array([point_values[label][i, :] for label in seg.m_tracking_markers])

            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(staticPos,dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt

                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)

            seg.getReferential("TF").addMotionFrame(csFrame)


        # --- LAJC
        desc = seg.getReferential('TF').static.getNode_byLabel(f"{prefix}AJC").m_desc
        values_AJCnode=seg.getReferential('TF').getNodeTrajectory(f"{prefix}AJC")

        values_AJCnode[invalid_indices] = np.zeros(3)

        btkTools.smartAppendPoint(aqui,f"{prefix}AJC",values_AJCnode, desc=str("opt"+desc))

        # --- KJC from Shank
        #values_KJCnode=seg.getReferential('TF').getNodeTrajectory("LKJC")
        #btkTools.smartAppendPoint(aqui,"LKJC-Shank",values_KJCnode, desc="opt from Shank")


        # remove KJC from list of tracking markers
        if f"{prefix}KJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove(f"{prefix}KJC")



    
    
    def _foot_motion_optimize(self,side:str, aqui: btk.btkAcquisition, dictRef: Dict[str, Any], motionMethod: enums.motionMethod) -> None:
        """Optimizes the motion of the foot segment using a specified motion method.

        This method adjusts the technical frame of the foot segment based on dynamic position data. It may add LAJC to the tracking markers if the marker list is less than 2 and checks the presence of tracking markers in the acquisition.

        Args:
            side (str): body side ( Left or Right)
            aqui (btk.btkAcquisition): Motion acquisition data.
            dictRef (Dict[str, Any]): Technical referential definitions for the foot.
            motionMethod (enums.motionMethod): Segmental motion method to apply.
        """
        prefix = "L" if side == "Left" else "R" if side == "Right" else ""

        seg=self.getSegment(f"{side} Foot")

        validFrames = btkTools.getValidFrames(aqui,seg.m_tracking_markers)
        seg.setExistFrames(validFrames)
        validFrames = np.array(validFrames)
        invalid_indices = ~validFrames

        #  --- add LAJC if marker list < 2  - check presence of tracking markers in the acquisition
        if seg.m_tracking_markers != []:
            if len(seg.m_tracking_markers)==2:
                if f"{prefix}AJC" not in seg.m_tracking_markers:
                    seg.m_tracking_markers.append(f"{prefix}AJC")
                    LOGGER.logger.debug(f"{prefix}AJC added to tracking marker list")

        # --- Motion of the Technical frame
        seg.getReferential("TF").motion =[]

        # part 1: get global location in Static
        if seg.m_tracking_markers != []: # work with tracking markers
            staticPos = np.array([seg.getReferential("TF").static.getNode_byLabel(label).m_global 
                                        for label in seg.m_tracking_markers])


        # part 2 : get dynamic position ( look out i pick up value in the btkAcquisition)
        point_values = {label: aqui.GetPoint(label).GetValues() for label in seg.m_tracking_markers}
        n_frames = aqui.GetPointFrameNumber()

        for i in range(0,n_frames):
            csFrame=frame.Frame()
            dynPos = np.array([point_values[label][i, :] for label in seg.m_tracking_markers])

            if motionMethod == enums.motionMethod.Sodervisk :
                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(staticPos,dynPos)
                R=np.dot(Ropt,seg.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,seg.getReferential("TF").static.getTranslation())+Lopt

                csFrame.setRotation(R)
                csFrame.setTranslation(tOri)

            seg.getReferential("TF").addMotionFrame(csFrame)


        # --- AJC from Foot
        values_AJCnode=seg.getReferential('TF').getNodeTrajectory(f"{prefix}AJC")
        values_AJCnode[invalid_indices] = np.zeros(3)
        
        # remove AJC from list of tracking markers
        if f"{prefix}AJC" in seg.m_tracking_markers: seg.m_tracking_markers.remove(f"{prefix}AJC")


    


    def _anatomical_motion(self, aqui: btk.btkAcquisition, segmentLabel: str, originLabel: str = "") -> None:
        """Computes the motion of an anatomical segment.

        This method adjusts the anatomical frame of a given segment based on the technical frame and relative anatomical matrix.

        Args:
            aqui (btk.btkAcquisition): Motion acquisition data.
            segmentLabel (str): Label of the segment for which the anatomical motion is being computed.
            originLabel (str, optional): Label of the origin point. Defaults to an empty string if not provided.
        """


        seg=self.getSegment(segmentLabel)

        # --- Motion of the Anatomical frame
        seg.anatomicalFrame.motion=[]

        ptOrigin=aqui.GetPoint(originLabel).GetValues()
        # computation
        
        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            csFrame.update(R,ptOrigin[i,:])
            seg.anatomicalFrame.addMotionFrame(csFrame)


    def _rotate_anatomical_motion(self, segmentLabel: str, angle: float, aqui: btk.btkAcquisition, options: Optional[Dict[str, Any]] = None) -> None:
        """Rotates the anatomical referential of a segment by a specified angle.

        This method applies a rotation to the anatomical frame of a specified segment. The rotation is defined by the given angle around the Z-axis of the anatomical frame.

        Args:
            segmentLabel (str): The label of the segment to be rotated.
            angle (float): The angle in degrees to rotate the anatomical frame.
            aqui (btk.btkAcquisition): Motion acquisition data.
            options (Optional[Dict[str, Any]]): Additional options (not used in this method).
        """


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
    def _rotateAjc(self, ajc: np.ndarray, kjc: np.ndarray, ank: np.ndarray, offset: float) -> np.ndarray:
        """Rotates the ankle joint center based on a specified offset.

        The method computes the rotation of the ankle joint center (AJC) around the abduction-adduction axis based on the provided offset. This rotation is applied to adjust the position of the AJC.

        Args:
            ajc (np.ndarray): The current position of the ankle joint center.
            kjc (np.ndarray): The knee joint center position.
            ank (np.ndarray): The ankle marker position.
            offset (float): The angle in degrees for the rotation.

        Returns:
            np.ndarray: The new position of the ankle joint center after rotation.
        """




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


    


    def _thorax_motion(self, aqui: btk.btkAcquisition, dictRef: dict, dictAnat: dict, options: Optional[dict] = None) -> None:
        """
        Computes the motion of the thorax segment based on biomechanical markers.

        This method calculates the technical and anatomical motion of the thorax segment using motion capture data. It involves the computation of joint centers and markers required for thorax motion analysis.

        Args:
            aqui (btk.btkAcquisition): The motion capture data.
            dictRef (dict): Dictionary containing technical referential definitions.
            dictAnat (dict): Dictionary containing anatomical referential definitions.
            options (Optional[dict]): Additional options for motion computation.
        """


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

        pt1=aqui.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][0])).GetValues()
        pt2=aqui.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][1])).GetValues()
        pt3=aqui.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][2])).GetValues()
        ptOrigin=aqui.GetPoint(str(dictRef["Thorax"]["TF"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)

        LSHO = aqui.GetPoint(str("LSHO")).GetValues()
        RSHO = aqui.GetPoint(str("RSHO")).GetValues()
        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Thorax"]["TF"]['sequence'])

        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.getReferential("TF").addMotionFrame(csFrame)

        axisX = np.vstack([obj.m_axisX for obj in seg.getReferential("TF").motion])
        OT = ptOrigin + -1.0*(markerDiameter/2.0)*axisX
        LVWM = np.cross((LSHO - OT ), axisX ) + LSHO
        RVWM = np.cross((RSHO - OT ), axisX ) + RSHO        

        for i in range(0,aqui.GetPointFrameNumber()):
            if validFrames[i]:
                OTvalues[i,:] = OT[i,:]
                LSJCvalues[i,:] = modelDecorator.VCMJointCentre( -1.0*(self.mp["LeftShoulderOffset"]+ markerDiameter/2.0) ,LSHO[i,:],OT[i,:],LVWM[i,:], beta=0 )
                LVWMvalues[i,:] = LVWM[i,:]
                RSJCvalues[i,:] = modelDecorator.VCMJointCentre( 1.0*(self.mp["RightShoulderOffset"]+ markerDiameter/2.0) ,RSHO[i,:],OT[i,:],RVWM[i,:], beta=0 )
                RVWMvalues[i,:] = RVWM[i,:]

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
        pt1=aqui.GetPoint(str(dictAnat["Thorax"]['labels'][0])).GetValues() #midTop
        pt2=aqui.GetPoint(str(dictAnat["Thorax"]['labels'][1])).GetValues() #midBottom
        pt3=aqui.GetPoint(str(dictAnat["Thorax"]['labels'][2])).GetValues() #midFront
        ptOrigin=aqui.GetPoint(str(dictAnat["Thorax"]['labels'][3])).GetValues() #OT

        C7 = aqui.GetPoint(str("C7")).GetValues()
        T10 = aqui.GetPoint(str("T10")).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
        x,y,z,R=frame.setFrameData(a1,a2,dictAnat["Thorax"]['sequence'])

        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.anatomicalFrame.addMotionFrame(csFrame)

            if hasattr(self,"_TopLumbar5"):
                T5inThorax[i,:] = np.dot(R[i].T,self._TopLumbar5[i,:]-ptOrigin[i,:])


            offset =( ptOrigin[i,:] + np.dot(R[i],np.array([-markerDiameter/2.0,0,0])) - ptOrigin[i,:])*1.05

            C7Global= C7[i,:] + offset
            C7inThorax[i,:] = np.dot(R[i].T,C7Global-ptOrigin[i,:])

            T10Global= T10[i,:] + offset
            T10inThorax[i,:] = np.dot(R[i].T,T10Global-ptOrigin[i,:])


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


    
    def _clavicle_motion(self, side: str, aqui: btk.btkAcquisition, dictRef: dict, dictAnat: dict, options: Optional[dict] = None) -> None:
        """
        Computes the motion of the clavicle segment for a specified side.

        This method is responsible for calculating the motion frames of both the technical and anatomical referentials of the clavicle, based on motion capture data.

        Args:
            side (str): Side of the body ('Left' or 'Right').
            aqui (btk.btkAcquisition): The motion capture data.
            dictRef (dict): Dictionary containing technical referential definitions.
            dictAnat (dict): Dictionary containing anatomical referential definitions.
            options (Optional[dict]): Additional options for motion computation.
        """


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

        pt1=aqui.GetPoint(str(dictRef[side+" Clavicle"]["TF"]['labels'][0])).GetValues()
        pt2=aqui.GetPoint(str(dictRef[side+" Clavicle"]["TF"]['labels'][1])).GetValues()
        pt3=aqui.GetPoint(str(dictRef[side+" Clavicle"]["TF"]['labels'][2])).GetValues()
        ptOrigin=aqui.GetPoint(str(dictRef[side+" Clavicle"]["TF"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)      
        x,y,z,R=frame.setFrameData(a1,a2,dictRef[side+" Clavicle"]["TF"]['sequence'])  

        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.getReferential("TF").addMotionFrame(csFrame)


        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA
        # computation
        pt1=aqui.GetPoint(str(dictAnat[side+" Clavicle"]['labels'][0])).GetValues()
        pt2=aqui.GetPoint(str(dictAnat[side+" Clavicle"]['labels'][1])).GetValues()
        pt3=aqui.GetPoint(str(dictAnat[side+" Clavicle"]['labels'][2])).GetValues()
        ptOrigin=aqui.GetPoint(str(dictAnat[side+" Clavicle"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
        x,y,z,R=frame.setFrameData(a1,a2,dictAnat[side+" Clavicle"]['sequence'])       
        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.anatomicalFrame.addMotionFrame(csFrame)

    
    def _upperArm_motion(self, side: str, aqui: btk.btkAcquisition, dictRef: dict, dictAnat: dict, options: Optional[dict] = None, frameReconstruction: str = "Both") -> None:
        """
        Computes the motion of the upper arm segment for a specified side.

        This function calculates motion frames for the upper arm segment, utilizing both technical and anatomical data from motion capture. It allows the option to reconstruct frames based on technical, anatomical, or both referentials.

        Args:
            side (str): Side of the body ('Left' or 'Right').
            aqui (btk.btkAcquisition): The motion capture data.
            dictRef (dict): Dictionary containing technical referential definitions.
            dictAnat (dict): Dictionary containing anatomical referential definitions.
            options (Optional[dict]): Additional options for motion computation.
            frameReconstruction (str): Specifies the type of frame reconstruction ('Technical', 'Anatomical', or 'Both').
        """



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

            pt1=aqui.GetPoint(str(dictRef[side+" UpperArm"]["TF"]['labels'][0])).GetValues()
            pt2=aqui.GetPoint(str(dictRef[side+" UpperArm"]["TF"]['labels'][1])).GetValues()
            pt3=aqui.GetPoint(str(dictRef[side+" UpperArm"]["TF"]['labels'][2])).GetValues()
            ptOrigin=aqui.GetPoint(str(dictRef[side+" UpperArm"]["TF"]['labels'][3])).GetValues()

            a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
            x,y,z,R=frame.setFrameData(a1,a2,dictRef[side+" UpperArm"]["TF"]['sequence'])

            for i in range(0,aqui.GetPointFrameNumber()):
                csFrame=frame.Frame()
                csFrame.setRotation(R[i])
                csFrame.setTranslation(ptOrigin[i,:])
                seg.getReferential("TF").addMotionFrame(csFrame)

                SJC = aqui.GetPoint(prefix+"SJC").GetValues()[i,:]
                LHE=aqui.GetPoint(prefix+"ELB").GetValues()[i,:]
                CVM = aqui.GetPoint(prefix+"CVM").GetValues()[i,:]

                #EJCvalues[i,:] =  modelDecorator.VCMJointCentre( (self.mp[side+"ElbowWidth"]+ markerDiameter)/2.0 ,LHE,SJC,CVM, beta=0 )
                if validFrames[i]:
                    EJCvalues[i,:] =  modelDecorator.VCMJointCentre( (self.mp[side+"ElbowWidth"]+ markerDiameter)/2.0 ,pt1[i,:],pt2[i,:],pt3[i,:], beta=0 )


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
            pt1=aqui.GetPoint(str(dictAnat[side+" UpperArm"]['labels'][0])).GetValues()
            pt2=aqui.GetPoint(str(dictAnat[side+" UpperArm"]['labels'][1])).GetValues()
            pt3=aqui.GetPoint(str(dictAnat[side+" UpperArm"]['labels'][2])).GetValues()
            ptOrigin=aqui.GetPoint(str(dictAnat[side+" UpperArm"]['labels'][3])).GetValues()

            a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
            x,y,z,R=frame.setFrameData(a1,a2,dictAnat[side+" UpperArm"]['sequence'])
            for i in range(0,aqui.GetPointFrameNumber()):
                csFrame=frame.Frame()
                csFrame.setRotation(R[i])
                csFrame.setTranslation(ptOrigin[i,:])
                seg.anatomicalFrame.addMotionFrame(csFrame)

    
    def _foreArm_motion(self, side: str, aqui: btk.btkAcquisition, dictRef: dict, dictAnat: dict, options: Optional[dict] = None, frameReconstruction: str = "both") -> None:
        """
        Computes the motion of the forearm segment for a specified side.

        This method calculates motion frames for the forearm segment, incorporating data from both technical and anatomical markers. It supports selecting between technical, anatomical, or both types of frame reconstruction.

        Args:
            side (str): Side of the body ('Left' or 'Right').
            aqui (btk.btkAcquisition): The motion capture data.
            dictRef (dict): Dictionary containing technical referential definitions.
            dictAnat (dict): Dictionary containing anatomical referential definitions.
            options (Optional[dict]): Additional options for motion computation.
            frameReconstruction (str): Type of frame reconstruction to use ('Technical', 'Anatomical', or 'Both').
        """


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

            pt1=aqui.GetPoint(str(dictRef[side+" ForeArm"]["TF"]['labels'][0])).GetValues()
            pt2=aqui.GetPoint(str(dictRef[side+" ForeArm"]["TF"]['labels'][1])).GetValues()
            pt3=aqui.GetPoint(str(dictRef[side+" ForeArm"]["TF"]['labels'][2])).GetValues()
            ptOrigin=aqui.GetPoint(str(dictRef[side+" ForeArm"]["TF"]['labels'][3])).GetValues()
            
            a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
            x,y,z,R=frame.setFrameData(a1,a2,dictRef[side+" ForeArm"]["TF"]['sequence'])
            for i in range(0,aqui.GetPointFrameNumber()):
                csFrame=frame.Frame()
                csFrame.setRotation(R[i])
                csFrame.setTranslation(ptOrigin[i,:])

                seg.getReferential("TF").addMotionFrame(csFrame)

                EJC = pt2[i,:]
                US=pt3[i,:]
                RS=pt1[i,:]

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

            pt1=aqui.GetPoint(str(dictAnat[side+" ForeArm"]['labels'][0])).GetValues()
            pt2=aqui.GetPoint(str(dictAnat[side+" ForeArm"]['labels'][1])).GetValues()
            if dictAnat[side+" ForeArm"]['labels'][2] is not None:
                pt3=aqui.GetPoint(str(dictAnat[side+" ForeArm"]['labels'][2])).GetValues()
            else:
                pt3 = None
                v = np.vstack([obj.m_axisY for obj in self.getSegment(side+" UpperArm").anatomicalFrame.motion])

            ptOrigin=aqui.GetPoint(str(dictAnat[side+" ForeArm"]['labels'][3])).GetValues()

            a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3,v=v)
            x,y,z,R=frame.setFrameData(a1,a2,dictAnat[side+" ForeArm"]['sequence'])
            # computation
            for i in range(0,aqui.GetPointFrameNumber()):
                csFrame=frame.Frame()
                csFrame.setRotation(R[i])
                csFrame.setTranslation(ptOrigin[i,:])

                seg.anatomicalFrame.addMotionFrame(csFrame)

    
    def _hand_motion(self, side: str, aqui: btk.btkAcquisition, dictRef: dict, dictAnat: dict, options: Optional[dict] = None) -> None:
        """
        Computes the motion of the hand segment for a specified side.

        This method is responsible for calculating motion frames of both the technical and anatomical referentials of the hand, based on motion capture data.

        Args:
            side (str): Side of the body ('Left' or 'Right').
            aqui (btk.btkAcquisition): The motion capture data.
            dictRef (dict): Dictionary containing technical referential definitions.
            dictAnat (dict): Dictionary containing anatomical referential definitions.
            options (Optional[dict]): Additional options for motion computation.
        """


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

        pt1=aqui.GetPoint(str(dictRef[side+" Hand"]["TF"]['labels'][0])).GetValues()
        pt2=aqui.GetPoint(str(dictRef[side+" Hand"]["TF"]['labels'][1])).GetValues()
        pt3=aqui.GetPoint(str(dictRef[side+" Hand"]["TF"]['labels'][2])).GetValues()
        ptOrigin=aqui.GetPoint(str(dictRef[side+" Hand"]["TF"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
        x,y,z,R=frame.setFrameData(a1,a2,dictRef[side+" Hand"]["TF"]['sequence'])
        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.getReferential("TF").addMotionFrame(csFrame)

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
        pt1=aqui.GetPoint(str(dictAnat[side+" Hand"]['labels'][0])).GetValues()
        pt2=aqui.GetPoint(str(dictAnat[side+" Hand"]['labels'][1])).GetValues()
        pt3=aqui.GetPoint(str(dictAnat[side+" Hand"]['labels'][2])).GetValues()
        ptOrigin=aqui.GetPoint(str(dictAnat[side+" Hand"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
        x,y,z,R=frame.setFrameData(a1,a2,dictAnat[side+" Hand"]['sequence'])
        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.anatomicalFrame.addMotionFrame(csFrame)

    
    def _head_motion(self, aqui: btk.btkAcquisition, dictRef: dict, dictAnat: dict, options: Optional[dict] = None) -> None:
        """
        Computes the motion of the head segment.

        This function calculates the motion frames for the head segment using both technical and anatomical data from motion capture. It involves the computation of necessary markers and joint centers for head motion analysis.

        Args:
            aqui (btk.btkAcquisition): The motion capture data.
            dictRef (dict): Dictionary containing technical referential definitions.
            dictAnat (dict): Dictionary containing anatomical referential definitions.
            options (Optional[dict]): Additional options for motion computation.
        """


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
        pt1=aqui.GetPoint(str(dictRef["Head"]["TF"]['labels'][0])).GetValues() 
        pt2=aqui.GetPoint(str(dictRef["Head"]["TF"]['labels'][1])).GetValues() 
        pt3=aqui.GetPoint(str(dictRef["Head"]["TF"]['labels'][2])).GetValues() 
        ptOrigin=aqui.GetPoint(str(dictRef["Head"]["TF"]['labels'][3])).GetValues()

        a1,a2 = self._frameByFrameAxesConstruction(pt1,pt2,pt3)
        x,y,z,R=frame.setFrameData(a1,a2,dictRef["Head"]["TF"]['sequence'])
        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()
            csFrame.setRotation(R[i])
            csFrame.setTranslation(ptOrigin[i,:])
            seg.getReferential("TF").addMotionFrame(csFrame)


        # --- motion of the anatomical referential
        seg.anatomicalFrame.motion=[]


        # additional markers
        # NA

        # computation
        ptOrigin=aqui.GetPoint(str(dictAnat["Head"]['labels'][3])).GetValues()
        for i in range(0,aqui.GetPointFrameNumber()):
            csFrame=frame.Frame()            
            R = np.dot(seg.getReferential("TF").motion[i].getRotation(), seg.getReferential("TF").relativeMatrixAnatomic)
            csFrame.update(R,ptOrigin[i,:])
            seg.anatomicalFrame.addMotionFrame(csFrame)


    # --- opensim --------
    def opensimTrackingMarkers(self) -> dict:
        """
        Retrieves tracking markers for OpenSim model configuration, excluding specific segments.

        This method provides a dictionary of tracking markers for segments, excluding those listed in the 'excluded' list. It is useful for configuring OpenSim models.

        Returns:
            dict: A dictionary with segment names as keys and corresponding tracking markers as values.
        """


        excluded = ["Thorax","Head","Left Clavicle", "Left UpperArm", "Left ForeArm",
                    "Left Hand", "Right Clavicle", "Right UpperArm", "Right ForeArm",
                     "Right Hand"]


        out={}
        for segIt in self.m_segmentCollection:
            if not segIt.m_isCloneOf and segIt.name not in excluded:
                out[segIt.name] = segIt.m_tracking_markers

        return out



    def opensimGeometry(self) -> dict:
        """
        Provides geometry configuration for OpenSim models.

        This method returns a dictionary containing the necessary information to configure the osim file for OpenSim. It includes joint labels and segment labels for key body parts.

        Returns:
            dict: A dictionary with configuration details for OpenSim geometry.
        """

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

    def opensimIkTask(self) -> dict:
        """
        Returns marker weights for Inverse Kinematics (IK) in OpenSim.

        This method provides a dictionary of markers and their respective weights, which are used in the IK process of OpenSim.

        Returns:
            dict: A dictionary where keys are marker names and values are their weights for IK.
        """


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
    def viconExport(self, NEXUS, acq:btk.btkAcquisition, vskName:str, pointSuffix:str, staticProcessingFlag:bool) -> None:
        """
        Exports model outputs to Nexus for Vicon systems.

        Args:
            NEXUS (viconnexus): Nexus handle for Vicon software integration.
            acq (btk.btkAcquisition): Acquisition data.
            vskName (str): Name of the VSK file.
            pointSuffix (str): Suffix for the points to be exported.
            staticProcessingFlag (bool): Flag to determine if only static model outputs will be exported.
        """


        pointSuffix  =  pointSuffix if pointSuffix is not None else ""

        if staticProcessingFlag:
            if self.checkCalibrationProperty("LeftKAD",True):
                nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"LKNE", acq)
            if self.checkCalibrationProperty("RightKAD",True):
                nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,"RKNE", acq)

        # export measured markers ( for CGM2.2 and 2.3)
        for it in btk.Iterate(acq.GetPoints()):
            if it.GetLabel()[-2:] == "_m":
                nexusTools.appendModelledMarkerFromAcq(NEXUS,vskName,it.GetLabel(),acq)


        # export Coms
        for it in btk.Iterate(acq.GetPoints()):
             if "Com_" in it.GetLabel():
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
