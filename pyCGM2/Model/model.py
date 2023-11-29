import numpy as np
import copy
import pyCGM2; LOGGER = pyCGM2.LOGGER
from typing import List, Tuple, Dict, Optional,Union,Any

import btk

from pyCGM2.Model import frame


from pyCGM2 import enums
from pyCGM2.Model import motion
from pyCGM2.Model.frame import Frame
from  pyCGM2.Tools import  btkTools
from pyCGM2.Math import  derivation
from  pyCGM2.Signal import signal_processing


class ClinicalDescriptor(object):
    """
    A clinical descriptor for biomechanical data analysis.

    Args:
        dataType (enums.DataType): Type of data (e.g., Angle, Moment, etc. See enums).
        jointOrSegmentLabel (str): Label of the joint or segment.
        indexes (List[int]): Indices of the outputs (usually a list of three elements).
        coefficients (List[float]): Coefficients to apply on the outputs.
        offsets (List[float]): Offsets to apply on the outputs (e.g., subtraction of 180 degrees).

    Kwargs:
        projection (Optional[enums.MomentProjection]): Coordinate system used to project the joint moment.

    Example:
        ```python
        ClinicalDescriptor(enums.DataType.Angle, "LHip", [0, 1, 2], [-1.0, -1.0, -1.0], [0.0, 0.0, 0.0])
        ```

    """
    def __init__(self, dataType: enums.DataType, jointOrSegmentLabel: str, indexes: List[int],
                 coefficients: List[float], offsets: List[float], **options: Dict[str, Any]):

        self.type = dataType
        self.label = jointOrSegmentLabel
        self.infos = {}
        self.infos["SaggitalIndex"] = indexes[0]
        self.infos["CoronalIndex"] = indexes[1]
        self.infos["TransversalIndex"] = indexes[2]
        self.infos["SaggitalCoeff"] = coefficients[0]
        self.infos["CoronalCoeff"] = coefficients[1]
        self.infos["TransversalCoeff"] = coefficients[2]
        self.infos["SaggitalOffset"] = offsets[0]
        self.infos["CoronalOffset"] = offsets[1]
        self.infos["TransversalOffset"] = offsets[2]

        if "projection" in options:
            self.projection = options["projection"]


class Joint(object):
    """
    A `Joint` is the connection between a proximal and a distal segment.

    Args:
        label (str): The label of the joint.
        proxLabel (str): The label of the proximal segment.
        distLabel (str): The label of the distal segment.
        sequence (str): The sequence of angles for the joint.
        nodeLabel (str): The label of the node associated with the joint.
    """

    def __init__(self, label: str, proxLabel: str, distLabel: str, sequence: str, nodeLabel: str):
 
        self.m_label=label
        self.m_proximalLabel=proxLabel
        self.m_distalLabel=distLabel
        self.m_sequence=sequence
        self.m_nodeLabel=nodeLabel
class Segment(object):
    """
    Represents a rigid body segment in a biomechanical model.

    A `Segment` is a fundamental component in biomechanical modeling, 
    representing a part of the body like a limb or a section of the spine. 
    It holds information about the segment's markers, technical and anatomical 
    referentials, body segment parameters (BSP), and other relevant biomechanical data.

    Args:
            label (str): Label of the segment.
            index (int): Unique index of the segment.
            sideEnum (enums.SegmentSide): Side of the body this segment belongs to.
            calibration_markers (List[str], optional): Labels of calibration markers associated with this segment. Defaults to [].
            tracking_markers (List[str], optional): Labels of tracking markers associated with this segment. Defaults to [].

    """
    ## TODO:
    # - compute constant matrix rotation between each referential.static



    def __init__(self, label: str, index: int, 
                 sideEnum:enums.SegmentSide, 
                 calibration_markers: Optional[List[str]] = [], 
                 tracking_markers: Optional[List[str]] = []):
        
        """Initializes a new instance of the Segment class"""
    
        self.name=label
        self.index=index
        self.side = sideEnum

        self.m_tracking_markers=tracking_markers
        self.m_calibration_markers = calibration_markers

        self.m_markerLabels=calibration_markers+tracking_markers
        self.referentials=[]
        self.anatomicalFrame =AnatomicalReferential()

        self.m_bsp = {}
        self.m_bsp["mass"] = 0
        self.m_bsp["length"] = 0
        self.m_bsp["rog"] = 0
        self.m_bsp["com"] = np.zeros((3))
        self.m_bsp["inertia"] = np.zeros((3,3))

        self.m_externalDeviceWrenchs = []
        self.m_externalDeviceBtkWrench = None
        self.m_proximalWrench = None

        self.m_proximalMomentContribution = {}
        self.m_proximalMomentContribution["internal"] = None
        self.m_proximalMomentContribution["external"] = None
        self.m_proximalMomentContribution["inertia"] = None
        self.m_proximalMomentContribution["linearAcceleration"] = None
        self.m_proximalMomentContribution["gravity"] = None
        self.m_proximalMomentContribution["externalDevices"] = None
        self.m_proximalMomentContribution["distalSegments"] = None
        self.m_proximalMomentContribution["distalSegmentForces"] = None
        self.m_proximalMomentContribution["distalSegmentMoments"] = None

        self.m_info = {}
        self.m_isCloneOf = False

        self.m_existFrames = None

    def setExistFrames(self, lstdata: List[int]):
        """
        Sets the frame numbers where the segment exists in the biomechanical model.

        Args:
            lstdata (List[int]): List of frame numbers where the segment is present.
        """
        self.m_existFrames = lstdata

    def getExistFrames(self) -> List[int]:
        """
        Returns the list of frame numbers where the segment exists.

        Returns:
            List[int]: Frame numbers where the segment is present.
        """
        return self.m_existFrames

    def removeTrackingMarker(self, labels: Union[str, List[str]]):
        """
        Remove one or multiple tracking markers from the segment.

        Args:
            labels (Union[str, List[str]]): Label or list of labels of the tracking markers to be removed.
        """

        if not isinstance(labels,list):
            labels = [labels]

        for label in labels:
            if label in self.m_tracking_markers:
                self.m_tracking_markers.remove(label)
                self.m_markerLabels.remove(label)
            else:
                LOGGER.logger.debug("tracking marker %s  remove" % label)


    def addTrackingMarkerLabel(self, labels: Union[str, List[str]]):
        """
        Add one or multiple tracking markers to the segment.

        Args:
            labels (Union[str, List[str]]): Label or list of labels of the tracking markers to be added.
        """

        if not isinstance(labels,list):
            labels = [labels]

        for label in labels:
            if label not in self.m_tracking_markers:
                self.m_tracking_markers.append(label)
                self.m_markerLabels.append(label)
            else:
                LOGGER.logger.debug("marker %s already in the tracking marker segment list" % label)

    def addCalibrationMarkerLabel(self, labels: Union[str, List[str]]):
        """
        Add one or multiple calibration markers to the segment.

        Args:
            labels (Union[str, List[str]]): Label or list of labels of the calibration markers to be added.
        """

        if not isinstance(labels,list):
            labels = [labels]

        for label in labels:
            if label not in self.m_calibration_markers:
                self.m_calibration_markers.append(label)
                self.m_markerLabels.append(label)
            else:
                LOGGER.logger.debug("marker %s already in the clibration marker segment list" % label)


    def resetMarkerLabels(self):
        """
        Reset the marker labels for the segment.
        """

        self.m_markerLabels = []
        self.m_markerLabels = self.m_tracking_markers + self.m_calibration_markers


    def addMarkerLabel(self, label: str):
        """
        Add a marker label to the segment.

        Args:
            label (str): Label of the marker to be added.
        """

        isFind=False
        i=0
        for marker in self.m_markerLabels:
            if label in marker:
                isFind=True
                index = i
            i+=1

        if isFind:
            LOGGER.logger.debug("marker %s already in the marker segment list" % label)


        else:
            self.m_markerLabels.append(label)

    def zeroingExternalDevice(self):
        """
        Zeroing (reset) the external device wrenches associated with the segment.
        """

        self.m_externalDeviceWrenchs = []
        self.m_externalDeviceBtkWrench = None

    def zeroingProximalWrench(self):
        """
        Zeroing (reset) the proximal wrench associated with the segment.
        """

        self.m_proximalWrench = None


    def downSampleExternalDeviceWrenchs(self, appf: int):
        """
        Downsample external device wrenches associated with the segment.

        Args:
            appf (int): Analog points per frame rate for downsampling.
        """
        
        if self.isExternalDeviceWrenchsConnected():

            for wrIt in  self.m_externalDeviceWrenchs:
                forceValues = wrIt.GetForce().GetValues()
                forceValues_ds= forceValues[::appf]
                wrIt.GetForce().SetValues(forceValues_ds)

                momentValues = wrIt.GetMoment().GetValues()
                momentValues_ds= momentValues[::appf]
                wrIt.GetMoment().SetValues(momentValues_ds)

                positionValues = wrIt.GetPosition().GetValues()
                positionValues_ds= positionValues[::appf]
                wrIt.GetPosition().SetValues(positionValues_ds)


    def isExternalDeviceWrenchsConnected(self) -> bool:
        """
        Check if any external device wrenches are connected to the segment.

        Returns:
            bool: True if external device wrenches are connected, False otherwise.
        """

        if self.m_externalDeviceWrenchs == []:
            return False
        else:
            return True

    def addExternalDeviceWrench(self, btkWrench: btk.btkWrench):
        """
        Add an external device wrench to the segment.

        Args:
            btkWrench (btk.btkWrench): A BTK wrench instance to be added.
        """
        self.m_externalDeviceWrenchs.append(btkWrench)


    def setMass(self, value: float):
        """
        Set the mass of the segment.

        Args:
            value (float): The length value to be set for the segment.
        """
        self.m_bsp["mass"] = value

    def setLength(self, value: float):
        """
        Set the length of the segment.

        Args:
            value (float): The length value to be set for the segment.
        """
        self.m_bsp["length"] = value

    def setRog(self, value: float):
        """
        Set the radius of gyration of the segment.

        Args:
            value (float): The radius of gyration value to be set for the segment.
        """
        self.m_bsp["rog"] = value

    def setComPosition(self, array3: np.ndarray):
        """
        Set the local position of the center of mass of the segment.

        Args:
            array3 (np.ndarray): A 3-element array representing the center of mass position.
        """
        self.m_bsp["com"] = array3

    def setInertiaTensor(self, array33: np.ndarray):
        """
        Set the inertia tensor of the segment.

        Args:
            array33 (np.ndarray): A 3x3 matrix representing the inertia tensor.
        """
        self.m_bsp["inertia"] = array33


    def addTechnicalReferential(self, label: str):
        """
        Add a technical referential to the segment.

        Args:
            label (str): The label of the technical referential to be added.
        """


        ref=TechnicalReferential(label)
        self.referentials.append(ref)



    def getReferential(self, label: str) -> 'TechnicalReferential':
        """
        Get a technical referential from the segment.

        Args:
            label (str): The label of the technical referential to retrieve.

        Returns:
            TechnicalReferential: The requested technical referential.
        """

        for tfIt in  self.referentials:
            if tfIt.label == label:
                return tfIt



    def getComTrajectory(self, exportBtkPoint: bool = False, btkAcq: Optional[btk.btkAcquisition] = None) -> np.ndarray:
        """
        Get the trajectory of the center of mass of the segment.

        Args:
            exportBtkPoint (bool, optional): If True, export as btk.point. Defaults to False.
            btkAcq (Optional[btk.btkAcquisition], optional): A btk acquisition instance if export is needed. Defaults to None.

        Returns:
            np.ndarray: An array representing the center of mass trajectory.
        """

        frameNumber = len(self.anatomicalFrame.motion)
        values = np.zeros((frameNumber,3))
        for i in range(0,frameNumber):
            values[i,:] = np.dot(self.anatomicalFrame.motion[i].getRotation() ,self.m_bsp["com"]) + self.anatomicalFrame.motion[i].getTranslation()

        if exportBtkPoint:
            if btkAcq != None:
                btkTools.smartAppendPoint(btkAcq,self.name + "_com",values,desc="com")
        return values


    def getComVelocity(self, pointFrequency: float, method: str = "spline") -> np.ndarray:
        """
        Calculate the linear velocity of the center of mass of the segment.

        Args:
            pointFrequency (float): The point frequency for the calculation.
            method (str, optional): The method for calculation ('spline' or 'spline fitting'). Defaults to 'spline'.

        Returns:
            np.ndarray: An array representing the center of mass linear velocity.
        """

        if method == "spline":
            values = derivation.splineDerivation(self.getComTrajectory(),pointFrequency,order=1)
        elif method == "spline fitting":
            values = derivation.splineFittingDerivation(self.getComTrajectory(),pointFrequency,order=1)
        else:
            values = derivation.firstOrderFiniteDifference(self.getComTrajectory(),pointFrequency)

        return values


    def getComAcceleration(self, pointFrequency: float, method: str = "spline", **options) -> np.ndarray:
        """
        Calculate the global linear acceleration of the center of mass of the segment.

        Args:
            pointFrequency (float): The point frequency for the calculation.
            method (str, optional): The method for calculation ('spline' or 'spline fitting'). Defaults to 'spline'.
            **options: Additional options for filtering.

        Returns:
            np.ndarray: An array representing the center of mass linear acceleration.
        """

        valueCom = self.getComTrajectory()
        if "fc" in options.keys() and  "order" in options.keys():
            valueCom = signal_processing.arrayLowPassFiltering(valueCom,pointFrequency,options["order"],options["fc"]  )

        if method == "spline":
            values = derivation.splineDerivation(valueCom,pointFrequency,order=2)
        elif method == "spline fitting":
            values = derivation.splineFittingDerivation(self.getComTrajectory(),pointFrequency,order=2)
        else:
            values = derivation.secondOrderFiniteDifference(valueCom,pointFrequency)

        return values





    def getAngularVelocity(self, sampleFrequency: float, method: str = "conventional") -> np.ndarray:
        """
        Calculate the angular velocity of the segment.

        Args:
            sampleFrequency (float): The sample frequency of the motion data.
            method (str, optional): The method used for computing the angular velocity ('conventional' or 'pig'). Defaults to 'conventional'.

        Returns:
            np.ndarray: An array representing the angular velocity of the segment.
        
        **Notes:**

        The *conventional* method computes angular velocity through the matrix product
        $\dot{R}R^t$

        The *pig* method duplicates a  bodybuilder code of the plug-in gait in
        which the velocity is computed from differentation between the next and previous pose
        """



        frameNumber = len(self.anatomicalFrame.motion)
        AngularVelocValues = np.zeros((frameNumber,3))

        # pig method0
        if method == "pig":
            for i in range(1,frameNumber-1):
                omegaX=(np.dot( self.anatomicalFrame.motion[i+1].m_axisY,
                                          self.anatomicalFrame.motion[i-1].m_axisZ))/(2*1/sampleFrequency)

                omegaY=(np.dot( self.anatomicalFrame.motion[i+1].m_axisZ,
                                          self.anatomicalFrame.motion[i-1].m_axisX))/(2*1/sampleFrequency)

                omegaZ=(np.dot( self.anatomicalFrame.motion[i+1].m_axisX,
                                self.anatomicalFrame.motion[i-1].m_axisY))/(2*1/sampleFrequency)

                omega = np.array([[omegaX,omegaY,omegaZ]]).transpose()

                AngularVelocValues[i,:] = np.dot(self.anatomicalFrame.motion[i].getRotation(),  omega).transpose()

        # conventional method
        if method == "conventional":
            rdot = derivation.matrixFirstDerivation(self.anatomicalFrame.motion, sampleFrequency)
            for i in range(1,frameNumber-1):
                tmp =np.dot(rdot[i],self.anatomicalFrame.motion[i].getRotation().transpose())
                AngularVelocValues[i,0]=tmp[2,1]
                AngularVelocValues[i,1]=tmp[0,2]
                AngularVelocValues[i,2]=tmp[1,0]

        return AngularVelocValues


    def getAngularAcceleration(self, sampleFrequency: float) -> np.ndarray:
        """
        Calculate the angular acceleration of the segment.

        Args:
            sampleFrequency (float): The sample frequency of the motion data.

        Returns:
            np.ndarray: An array representing the angular acceleration of the segment.
        """

        values = derivation.firstOrderFiniteDifference(self.getAngularVelocity(sampleFrequency),sampleFrequency)
        return values

# -------- ABSTRACT MODEL ---------

class Model(object):
    """
    Base class representing a biomechanical model.

    A `Model` consists of segments, joints, and body segment parameters.
    """

    def __init__(self):

        self.m_segmentCollection=[]
        self.m_jointCollection=[]
        self.mp={}
        self.mp_computed={}
        self.m_chains={}
        self.m_staticFilename=None

        self.m_properties={}
        self.m_properties["CalibrationParameters"]={}
        self.m_clinicalDescriptors= []
        self.m_csDefinitions = []
        self.m_bodypart=None
        self.m_centreOfMass=None

    def __repr__(self):
        return "Basis Model"

    def setBodyPart(self, bodypart: str):
        """ 
        [Obsolete] Specify which body part is represented by the model.
        Args:
            bodypart (str): The body part represented by the model.
        """
        self.m_bodypart = bodypart

    def getBodyPart(self) -> str:
        """ 
        [Obsolete] Return the body part represented by the model.
        Returns:
            str: The body part represented by the model.
        """
        return self.m_bodypart

    def setCentreOfMass(self, com: np.ndarray):
        """
        Set the center of mass trajectory.
        Args:
            com (np.ndarray): An array (n, 3) representing the center of mass trajectory.
        """
        self.m_centreOfMass = com

    def getCentreOfMass(self):
        """
        Return the center of mass trajectory.
        Returns:
            np.ndarray: The center of mass trajectory.
        """
        return self.m_centreOfMass

    def setProperty(self, propertyLabel: str, value: Any):
        """
        Set or update a property in the property dictionary.
        Args:
            propertyLabel (str): The property label.
            value (Any): The property value.
        """
        self.m_properties[propertyLabel] = value

    def getProperty(self, propertyLabel: str) -> Any:
        """
        Return a specified property.
        Args:
            propertyLabel (str): The property label.
        Returns:
            Any: The value of the specified property.
        """
        try:
            return self.m_properties[propertyLabel]
        except:
            raise ("property Label doesn t find")

    def setCalibrationProperty(self, propertyLabel: str, value: Any):
        """
        Set or update a calibration property in the property dictionary.
        Args:
            propertyLabel (str): The property label.
            value (Any): The property value.
        """
        self.m_properties["CalibrationParameters"][propertyLabel] = value

    def isProperty(self, label: str) -> bool:
        """
        Check if a property exists by its label.
        Args:
            label (str): The property label.
        Returns:
            bool: True if the property exists, False otherwise.
        """
        return True if label in self.m_properties.keys() else False

    def isCalibrationProperty(self, label: str) -> bool:
        """
        Check if a calibration property exists by its label.
        Args:
            label (str): The property label.
        Returns:
            bool: True if the calibration property exists, False otherwise.
        """
        return True if label in self.m_properties["CalibrationParameters"].keys() else False

    def checkCalibrationProperty(self, CalibrationParameterLabel: str, value: Any) -> bool:
        """
        Check if a calibration property matches a specific value.
        Args:
            CalibrationParameterLabel (str): The calibration parameter label.
            value (Any): The value to compare with the calibration property.
        Returns:
            bool: True if the calibration property matches the specified value, False otherwise.
        """
        if self.isCalibrationProperty(CalibrationParameterLabel):
            if self.m_properties["CalibrationParameters"][CalibrationParameterLabel] == value:
                return True
            else:
                return False
        else:
            return False
            LOGGER.logger.warning("[pyCGM2] : CalibrationParameterLabel doesn t exist")

    def setStaticFilename(self, name: str):
        """
        Set the static filename used for static calibration.
        Args:
            name (str): The filename.
        """
        self.m_staticFilename=name

    def addChain(self, label: str, indexSegmentList: List[int]):
        """
        Add a segment chain to the model.
        Args:
            label (str): Label of the chain.
            indexSegmentList (List[int]): Indexes of the segments constituting the chain.
        """
        self.m_chains[label] = indexSegmentList

    def addJoint(self, label: str, proxLabel: str, distLabel: str, sequence: str, nodeLabel: str):
        """
        Add a joint to the model.
        Args:
            label (str): Label of the joint.
            proxLabel (str): Label of the proximal segment.
            distLabel (str): Label of the distal segment.
            sequence (str): Sequence angle.
            nodeLabel (str): Node label.
        """

        j=Joint( label, proxLabel,distLabel,sequence,nodeLabel)
        self.m_jointCollection.append(j)


    def addSegment(self, label: str, index: int, sideEnum: enums.SegmentSide, calibration_markers: List[str] = [], tracking_markers: List[str] = [], cloneOf: bool = False):
        """
        Add a segment to the model.
        Args:
            label (str): Label of the segment.
            index (int): Index of the segment.
            sideEnum (enums.SegmentSide): Body side of the segment.
            calibration_markers (List[str], optional): Labels of the calibration markers. Defaults to [].
            tracking_markers (List[str], optional): Labels of the tracking markers. Defaults to [].
            cloneOf (bool, optional): Indicates if the segment is a clone of another. Defaults to False.
        """

        seg=Segment(label,index,sideEnum,calibration_markers,tracking_markers)
        if cloneOf is True:
            seg.m_isCloneOf = True
        self.m_segmentCollection.append(seg)


    def updateSegmentFromCopy(self, targetLabel: str, segmentToCopy: Segment):
        """
        Update a segment from a copy of another segment instance.
        Args:
            targetLabel (str): Label of the segment to be updated.
            segmentToCopy (Segment): A `Segment` instance to copy from.
        """
        copiedSegment = copy.deepcopy(segmentToCopy)
        copiedSegment.name = targetLabel

        isClone = True if self.getSegment(targetLabel).m_isCloneOf else False
        for i in range(0, len(self.m_segmentCollection)):
            if self.m_segmentCollection[i].name == targetLabel:
                self.m_segmentCollection[i] = copiedSegment
                self.m_segmentCollection[i].m_isCloneOf = isClone


    def removeSegment(self, segmentLabels: List[str]):
        """
        Remove `Segment` instances based on their labels.
        Args:
            segmentLabels (List[str]): List of segment labels to be removed.
        """

        segment_list = [it for it in self.m_segmentCollection if it.name not in segmentLabels]
        self.m_segmentCollection = segment_list

    def removeJoint(self, jointLabels: List[str]):
        """
        Remove `Joint` instances based on their labels.
        Args:
            jointLabels (List[str]): List of joint labels to be removed.
        """
        joint_list = [it for it in self.m_jointCollection if it.m_label not in jointLabels]
        self.m_jointCollection = joint_list



    def getSegment(self, label: str):
        """
        Retrieve a `Segment` instance based on its label.
        Args:
            label (str): Label of the Segment.
        Returns:
            Optional[Segment]: The Segment instance if found, None otherwise.
        """

        for it in self.m_segmentCollection:
            if it.name == label:
                return it

    def getSegmentIndex(self, label: str) -> Optional[int]:
        """
        Retrieve the index of a `Segment` based on its label.
        Args:
            label (str): Label of the Segment.
        Returns:
            Optional[int]: The index of the Segment if found, None otherwise.
        """
        index=0
        for it in self.m_segmentCollection:
            if it.name == label:
                return index
            index+=1


    def getSegmentByIndex(self, index: int) -> Optional[Segment]:
        """
        Retrieve a `Segment` instance based on its index.
        Args:
            index (int): Index of the Segment.
        Returns:
            Optional[Segment]: The Segment instance if found, None otherwise.
        """

        for it in self.m_segmentCollection:
            if it.index == index:
                return it

    def getSegmentList(self) -> List[str]:
        """
        Retrieve a list of all segment labels.
        Returns:
            List[str]: List of segment labels.
        """
        return [it.name for it in self.m_segmentCollection]


    def getJointList(self) -> List[str]:
        """
        Retrieve a list of all joint labels.
        Returns:
            List[str]: List of joint labels.
        """
        return [it.m_label for it in self.m_jointCollection]



    def getJoint(self, label: str) -> Optional[Joint]:
        """
        Retrieve a `Joint` instance based on its label.
        Args:
            label (str): Label of the joint.
        Returns:
            Optional[Joint]: The Joint instance if found, None otherwise.
        """

        for it in self.m_jointCollection:
            if it.m_label == label:
                return it

    def addAnthropoInputParameters(self, iDict: Dict[str, Any], optional: Optional[Dict[str, Any]] = None):
        """
        Add measured anthropometric data to the model.
        Args:
            iDict (Dict[str, Any]): Required anthropometric data.
            optional (Optional[Dict[str, Any]]): Optional anthropometric data.
        """

        self.mp=iDict

        if optional is not None:
            self.mp.update(optional)


    def decomposeTrackingMarkers(self, acq: btk.btkAcquisition, TechnicalFrameLabel: str):
        """
        Decompose tracking markers to their components.
        Args:
            acq (btk.btkAcquisition): BTK acquisition instance.
            TechnicalFrameLabel (str): Label of the technical frame.
        """

        for seg in self.m_segmentCollection:

            for marker in seg.m_tracking_markers:

                nodeTraj= seg.getReferential(TechnicalFrameLabel).getNodeTrajectory(marker)
                markersTraj =acq.GetPoint(marker).GetValues()

                markerTrajectoryX=np.array( [ markersTraj[:,0], nodeTraj[:,1], nodeTraj[:,2]]).T
                markerTrajectoryY=np.array( [ nodeTraj[:,0], markersTraj[:,1], nodeTraj[:,2]]).T
                markerTrajectoryZ=np.array( [ nodeTraj[:,0], nodeTraj[:,1], markersTraj[:,2]]).T


                btkTools.smartAppendPoint(acq,marker+"-X",markerTrajectoryX,PointType="Marker", desc="")
                btkTools.smartAppendPoint(acq,marker+"-Y",markerTrajectoryY,PointType="Marker", desc="")
                btkTools.smartAppendPoint(acq,marker+"-Z",markerTrajectoryZ,PointType="Marker", desc="")


    def displayStaticCoordinateSystem(self, aquiStatic: btk.btkAcquisition, 
                                      segmentLabel: str, targetPointLabel: str, referential: str = "Anatomic"):
        """
        Display a static coordinate system.
        Args:
            aquiStatic (btk.btkAcquisition): BTK acquisition instance from a static C3D.
            segmentLabel (str): Segment label.
            targetPointLabel (str): Label of the point defining axis limits.
            referential (str, Optional): Type of segment coordinate system to display (default is "Anatomic").


        """

        seg=self.getSegment(segmentLabel)
        if referential == "Anatomic":
            ref =seg.anatomicalFrame
        else:
            ref = seg.getReferential("TF")

        val =  np.dot(ref.static.getRotation() , np.array([100.0,0.0,0.0])) + ref.static.getTranslation()
        btkTools.smartAppendPoint(aquiStatic,targetPointLabel+"_X",val*np.ones((aquiStatic.GetPointFrameNumber(),3)),desc="")
        val =  np.dot(ref.static.getRotation() , np.array([0.0,100.0,0.0])) + ref.static.getTranslation()
        btkTools.smartAppendPoint(aquiStatic,targetPointLabel+"_Y",val*np.ones((aquiStatic.GetPointFrameNumber(),3)),desc="")
        val =  np.dot(ref.static.getRotation() , np.array([0.0,0,100.0])) + ref.static.getTranslation()
        btkTools.smartAppendPoint(aquiStatic,targetPointLabel+"_Z",val*np.ones((aquiStatic.GetPointFrameNumber(),3)),desc="")

    def displayMotionCoordinateSystem(self, acqui: btk.btkAcquisition, segmentLabel: str, targetPointLabel: str, referential: str = "Anatomic"):
        """
        Display a motion coordinate system.
        Args:
            acqui (btk.btkAcquisition): BTK acquisition instance.
            segmentLabel (str): Segment label.
            targetPointLabel (str): Label of the point defining axis limits.
            referential (str, Optional): Type of segment coordinate system to display (default is "Anatomic").
        """
        seg=self.getSegment(segmentLabel)
        valX=np.zeros((acqui.GetPointFrameNumber(),3))
        valY=np.zeros((acqui.GetPointFrameNumber(),3))
        valZ=np.zeros((acqui.GetPointFrameNumber(),3))


        if referential == "Anatomic":
            ref =seg.anatomicalFrame
        else:
            ref = seg.getReferential("TF")

        for i in range(0,acqui.GetPointFrameNumber()):
            valX[i,:]= np.dot(ref.motion[i].getRotation() , np.array([100.0,0.0,0.0])) + ref.motion[i].getTranslation()
            valY[i,:]= np.dot(ref.motion[i].getRotation() , np.array([0.0,100.0,0.0])) + ref.motion[i].getTranslation()
            valZ[i,:]= np.dot(ref.motion[i].getRotation() , np.array([0.0,0.0,100.0])) + ref.motion[i].getTranslation()

        btkTools.smartAppendPoint(acqui,targetPointLabel+"_X",valX,desc="")
        btkTools.smartAppendPoint(acqui,targetPointLabel+"_Y",valY,desc="")
        btkTools.smartAppendPoint(acqui,targetPointLabel+"_Z",valZ,desc="")

    def displayMotionViconCoordinateSystem(self, acqui: btk.btkAcquisition, segmentLabel: str, targetPointLabelO: str, targetPointLabelX: str, targetPointLabelY: str, targetPointLabelZ: str, referential: str = "Anatomic"):
        """
        Display a motion Vicon coordinate system.
        Args:
            acqui (btk.btkAcquisition): BTK acquisition instance.
            segmentLabel (str): Segment label.
            targetPointLabelO (str): Label for the origin point.
            targetPointLabelX (str): Label for the X-axis point.
            targetPointLabelY (str): Label for the Y-axis point.
            targetPointLabelZ (str): Label for the Z-axis point.
            referential (str, Optional): Type of segment coordinate system to display (default is "Anatomic").
        """

        seg=self.getSegment(segmentLabel)

        origin=np.zeros((acqui.GetPointFrameNumber(),3))
        valX=np.zeros((acqui.GetPointFrameNumber(),3))
        valY=np.zeros((acqui.GetPointFrameNumber(),3))
        valZ=np.zeros((acqui.GetPointFrameNumber(),3))


        if referential == "Anatomic":
            ref =seg.anatomicalFrame
        else:
            ref = seg.getReferential("TF")

        for i in range(0,acqui.GetPointFrameNumber()):
            origin[i,:] = ref.motion[i].getTranslation()
            valX[i,:]= np.dot(ref.motion[i].getRotation() , np.array([100.0,0.0,0.0])) + ref.motion[i].getTranslation()
            valY[i,:]= np.dot(ref.motion[i].getRotation() , np.array([0.0,100.0,0.0])) + ref.motion[i].getTranslation()
            valZ[i,:]= np.dot(ref.motion[i].getRotation() , np.array([0.0,0.0,100.0])) + ref.motion[i].getTranslation()

        btkTools.smartAppendPoint(acqui,targetPointLabelO,origin,desc="")
        btkTools.smartAppendPoint(acqui,targetPointLabelX,valX,desc="")
        btkTools.smartAppendPoint(acqui,targetPointLabelY,valY,desc="")
        btkTools.smartAppendPoint(acqui,targetPointLabelZ,valZ,desc="")

    def setClinicalDescriptor(self, jointOrSegmentLabel: str, dataType: enums.DataType, indexes: List[int], coefficients: List[float], offsets: List[float], **options):
        """
        Set a clinical descriptor.
        Args:
            jointOrSegmentLabel (str): Segment or joint label.
            dataType (enums.DataType): Data type.
            indexes (List[int]): Indexes.
            coefficients (List[float]): Coefficients to apply on outputs.
            offsets (List[float]): Offsets to apply on outputs.
        
        Kwargs:
            projection (enums.MomentProjection): Coordinate system used to project the joint moment.

        ```python
            model.setClinicalDescriptor("LHip",enums.DataType.Angle, [0,1,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0]
        ```

        """

        descriptor = ClinicalDescriptor(dataType, jointOrSegmentLabel, indexes,coefficients, offsets,**options)
        self.m_clinicalDescriptors.append(descriptor)

    def getClinicalDescriptor(self, dataType: enums.DataType, jointOrSegmentLabel: str, projection: Optional[enums.MomentProjection] = None) -> Optional[Dict[str, Any]]:
        """
        Return a clinical descriptor.
        Args:
            dataType (enums.DataType): Data type.
            jointOrSegmentLabel (str): Segment or joint label.
            projection (Optional[enums.MomentProjection]): Joint moment projection.
        Returns:
            Optional[Dict[str, Any]]: Clinical descriptor if found, None otherwise.

        """

        if self.m_clinicalDescriptors !=[]:
            for descriptor in self.m_clinicalDescriptors:
                if projection is None:
                    if descriptor.type == dataType and descriptor.label ==jointOrSegmentLabel:
                        infos= descriptor.infos
                        break
                    else:
                        infos = False
                else:
                    if descriptor.type == dataType and descriptor.label ==jointOrSegmentLabel and descriptor.projection == projection:
                        infos= descriptor.infos
                        break
                    else:
                        infos = False
        else:
            infos=False

        if not infos:
            LOGGER.logger.debug("[pyCGM2] : descriptor [ type: %s - label: %s]  not found" %(dataType.name,jointOrSegmentLabel))

        return infos

    def setCoordinateSystemDefinition(self, segmentLabel: str, coordinateSystemLabel: str, referentialType: str):
        """
        Set coordinate system definition.
        Args:
            segmentLabel (str): Segment label.
            coordinateSystemLabel (str): Coordinate system label.
            referentialType (str): Type of referential.
        """
        dic = {"segmentLabel": segmentLabel,"coordinateSystemLabel": coordinateSystemLabel,"referentialType": referentialType}
        self.m_csDefinitions.append(dic)


class Model6Dof(Model):
    """The Model6Dof class, inheriting from Model, specifically deals with models 
    having six degrees of freedom."""

    def __init__(self):
        super(Model6Dof, self).__init__()

    def _calibrateTechnicalSegment(self, aquiStatic: btk.btkAcquisition, segName: str, dictRef: Dict[str, Any], frameInit: int, frameEnd: int, options: Optional[Dict[str, Any]] = None):
        """
        Calibrate technical segment.
        Args:
            aquiStatic (btk.btkAcquisition): Static acquisition data.
            segName (str): Name of the segment.
            dictRef (Dict[str, Any]): Reference dictionary.
            frameInit (int): Initial frame number.
            frameEnd (int): Ending frame number.
            options (Optional[Dict[str, Any]]): Additional options.
        """

        segPicked=self.getSegment(segName)
        for tfName in dictRef[segName]: # TF name

            pt1=aquiStatic.GetPoint(str(dictRef[segName][tfName]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            pt2=aquiStatic.GetPoint(str(dictRef[segName][tfName]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
            pt3=aquiStatic.GetPoint(str(dictRef[segName][tfName]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

            ptOrigin=aquiStatic.GetPoint(str(dictRef[segName][tfName]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)


            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictRef[segName][tfName]['sequence'])

            segPicked.referentials[-1].static.m_axisX=x # work on the last TF in the list : thus index -1
            segPicked.referentials[-1].static.m_axisY=y
            segPicked.referentials[-1].static.m_axisZ=z

            segPicked.referentials[-1].static.setRotation(R)
            segPicked.referentials[-1].static.setTranslation(ptOrigin)

            #  - add Nodes in segmental static(technical)Frame -
            for label in segPicked.m_markerLabels:
                globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
                segPicked.referentials[-1].static.addNode(label,globalPosition,positionType="Global")

    def _calibrateAnatomicalSegment(self, aquiStatic: btk.btkAcquisition,
                                      segName: str, dictAnatomic: Dict[str, Any], 
                                      frameInit: int, frameEnd: int, 
                                      options: Optional[Dict[str, Any]] = None):
        """
        Calibrate anatomical segment.
        Args:
            aquiStatic (btk.btkAcquisition): Static acquisition data.
            segName (str): Name of the segment.
            dictAnatomic (Dict[str, Any]): Anatomic dictionary.
            frameInit (int): Initial frame number.
            frameEnd (int): Ending frame number.
            options (Optional[Dict[str, Any]]): Additional options.
        """
            
        # calibration of technical Frames
        for segName in dictAnatomic:

            segPicked=self.getSegment(segName)
            tf=segPicked.getReferential("TF")

            nd1 = str(dictAnatomic[segName]['labels'][0])
            pt1 = tf.static.getNode_byLabel(nd1).m_global

            nd2 = str(dictAnatomic[segName]['labels'][1])
            pt2 = tf.static.getNode_byLabel(nd2).m_global

            nd3 = str(dictAnatomic[segName]['labels'][2])
            pt3 = tf.static.getNode_byLabel(nd3).m_global

            ndO = str(dictAnatomic[segName]['labels'][3])
            ptO = tf.static.getNode_byLabel(ndO).m_global

            a1=(pt2-pt1)
            a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

            v=(pt3-pt1)
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            a2=np.cross(a1,v)
            a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

            x,y,z,R=frame.setFrameData(a1,a2,dictAnatomic[segName]['sequence'])

            segPicked.anatomicalFrame.static.m_axisX=x # work on the last TF in the list : thus index -1
            segPicked.anatomicalFrame.static.m_axisY=y
            segPicked.anatomicalFrame.static.m_axisZ=z

            segPicked.anatomicalFrame.static.setRotation(R)
            segPicked.anatomicalFrame.static.setTranslation(ptO)

            # --- relative rotation Technical Anatomical
            tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,segPicked.anatomicalFrame.static.getRotation()))

            # add tracking markers as node
            for label in segPicked.m_markerLabels:
                globalPosition=aquiStatic.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
                segPicked.anatomicalFrame.static.addNode(label,globalPosition,positionType="Global")


    def computeMotionTechnicalFrame(self, aqui: btk.btkAcquisition, segName: str, dictRef: Dict[str, Any], method: enums.motionMethod, options: Optional[Dict[str, Any]] = None):
        """
        Compute motion for the technical frame.
        Args:
            aqui (btk.btkAcquisition): Dynamic acquisition data.
            segName (str): Name of the segment.
            dictRef (Dict[str, Any]): Reference dictionary.
            method (enums.motionMethod): Method for motion computation.
            options (Optional[Dict[str, Any]]): Additional options.
        """

        segPicked=self.getSegment(segName)
        segPicked.getReferential("TF").motion =[]
        if method == enums.motionMethod.Sodervisk :
            tms= segPicked.m_tracking_markers
            for i in range(0,aqui.GetPointFrameNumber()):
                visibleMarkers = btkTools.getVisibleMarkersAtFrame(aqui,tms,i)

                # constructuion of the input of sodervisk
                arrayStatic = np.zeros((len(visibleMarkers),3))
                arrayDynamic = np.zeros((len(visibleMarkers),3))

                j=0
                for vm in visibleMarkers:
                    arrayStatic[j,:] = segPicked.getReferential("TF").static.getNode_byLabel(vm).m_global
                    arrayDynamic[j,:] = aqui.GetPoint(vm).GetValues()[i,:]
                    j+=1

                Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(arrayStatic,arrayDynamic)
                R=np.dot(Ropt,segPicked.getReferential("TF").static.getRotation())
                tOri=np.dot(Ropt,segPicked.getReferential("TF").static.getTranslation())+Lopt

                cframe=frame.Frame()
                cframe.setRotation(R)
                cframe.setTranslation(tOri)
                cframe.m_axisX=R[:,0]
                cframe.m_axisY=R[:,1]
                cframe.m_axisZ=R[:,2]

                segPicked.getReferential("TF").addMotionFrame(copy.deepcopy(cframe) )
        else:
            raise Exception("[pyCGM2] : motion method doesn t exist")

    def computeMotionAnatomicalFrame(self, aqui: btk.btkAcquisition, segName: str, dictAnatomic: Dict[str, Any], options: Optional[Dict[str, Any]] = None):
        """
        Compute motion for the anatomical frame.
        Args:
            aqui (btk.btkAcquisition): Dynamic acquisition data.
            segName (str): Name of the segment.
            dictAnatomic (Dict[str, Any]): Anatomic dictionary.
            options (Optional[Dict[str, Any]]): Additional options.
        """

        segPicked=self.getSegment(segName)

        segPicked.anatomicalFrame.motion=[]

        ndO = str(dictAnatomic[segName]['labels'][3])
        ptO = segPicked.getReferential("TF").getNodeTrajectory(ndO)

        csFrame=frame.Frame()
        for i in range(0,aqui.GetPointFrameNumber()):
            R = np.dot(segPicked.getReferential("TF").motion[i].getRotation(), segPicked.getReferential("TF").relativeMatrixAnatomic)
            csFrame.update(R,ptO)
            segPicked.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))


# --------  MODEL COMPONANTS ---------
class Referential(object):
    """
    Represents a segmental coordinate system.

    Attributes:
        static (Frame): A Frame instance characterizing the mean pose from a static trial.
        motion (List[Frame]): A list of Frame instances characterizing the pose at each time-frame of a dynamic trial.
        relativeMatrixAnatomic (numpy.array): A matrix representing the relative rotation of the anatomical Referential expressed in the technical referential.
        additionalInfos (Dict): Additional information about the referential.


    """
    def __init__(self):
        self.static=frame.Frame()
        self.motion=[]
        self.relativeMatrixAnatomic = np.zeros((3,3))
        self.additionalInfos = {}

    def setStaticFrame(self, frame: Frame):
        """
        Set the static pose of the referential.
        Args:
            frame (Frame): A Frame instance to set as the static pose.
        """
        self.static = frame


    def addMotionFrame(self, frame: Frame):
        """
        Append a Frame instance to the motion attribute.
        Args:
            frame (Frame): A Frame instance representing a pose at a specific time frame.
        """
        self.motion.append(frame)

    def getNodeTrajectory(self, label: str) -> np.ndarray:
        """
        Return the trajectory of a node.
        Args:
            label (str): Label of the desired node.
        Returns:
            np.ndarray: Trajectory of the specified node.
        """

        node=self.static.getNode_byLabel(label)
        nFrames=len(self.motion)
        pt=np.zeros((nFrames,3))
        for i in range(0,nFrames):
            pt[i,:]=np.dot(self.motion[i].getRotation(),node.m_local)+ self.motion[i].getTranslation()

        return pt

    def getOriginTrajectory(self) -> np.ndarray:
        """
        Return the trajectory of the origin of the referential.
        Returns:
            np.ndarray: Trajectory of the origin.
        """
        nFrames=len(self.motion)
        pt=np.zeros((nFrames,3))
        for i in range(0,nFrames):
            pt[i,:]=self.motion[i].getTranslation()

        return pt

    def getAxisTrajectory(self, axis: str) -> np.ndarray:
        """
        Return the trajectory of a specified axis.
        Args:
            axis (str): Axis for which the trajectory is required. Must be 'X', 'Y', or 'Z'.
        Returns:
            np.ndarray: Trajectory of the specified axis.
        """
        nFrames=len(self.motion)
        pt=np.zeros((nFrames,3))
        for i in range(0,nFrames):
            if axis == "X":
                pt[i,:]=self.motion[i].m_axisX
            elif axis == "Y":
                pt[i,:]=self.motion[i].m_axisY
            elif axis == "Z":
                pt[i,:]=self.motion[i].m_axisZ
            else:
                raise Exception("[pyCGM2] axis cannot be different than X, Y, or Z")

        return pt

class TechnicalReferential(Referential):
    """
    Represents a Technical Referential (Coordinate System) constructed from tracking markers.

    This class extends the `Referential` class and is specialized for technical referentials 
    constructed from tracking markers of a segment during motion capture.

    Attributes:
        label (str): Label of the technical referential.
        relativeMatrixAnatomic (numpy.array): Matrix representing the relative rotation of the anatomical Referential 
                                              expressed in this technical referential.
    Args:
        label (str): Label of the technical referential.
    """

    def __init__(self, label:str):

        super(TechnicalReferential, self).__init__()

        self.label=label
        self.relativeMatrixAnatomic=np.eye(3,3)

    def setRelativeMatrixAnatomic(self, array: np.ndarray):
        """
        Set the relative rigid rotation of the anatomical Referential expressed in the technical referential.

        This matrix transformation is used to relate the technical referential to the anatomical referential,
        providing a means to convert between the two coordinate systems.

        Args:
            array (np.ndarray): A 3x3 numpy array representing the rigid rotation matrix.
        """
        self.relativeMatrixAnatomic = array

class AnatomicalReferential(Referential):
    """
    Represents an Anatomical Referential (Coordinate System) constructed for a segment.

    This class extends the `Referential` class and is specialized for anatomical referentials 
    which are typically defined during a static pose with calibration markers. 
    It provides the means to represent and manipulate anatomical coordinate systems in a biomechanical model.

    """
    def __init__(self):
        super(AnatomicalReferential, self).__init__()




