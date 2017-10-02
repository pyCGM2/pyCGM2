# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 14:27:51 2015
@author: fleboeuf
"""

import numpy as np
import btk
import copy
import logging

import frame

from  pyCGM2.Tools import  btkTools
from pyCGM2.Math import  derivation

from  pyCGM2.Signal import signal_processing



# -------- ABSTRACT MODEL ---------

class Model(object):
    """
        Abstract class `Model`.
        `Model` collects *segments*, *joints*, *body segment parameters* and present convenient model-componant accessors


    """

    def __init__(self):

        self.m_segmentCollection=[]
        self.m_jointCollection=[]
        self.mp=dict()
        self.mp_computed=dict()
        self.m_chains=dict()
        self.m_staticFilename=None

        self.m_properties=dict()
        self.m_properties["CalibrationParameters"]=dict()

    def __repr__(self):
        return "Basis Model"

    def setProperty(self, propertyLabel,  value):
        self.m_properties[propertyLabel] = value

    def setCalibrationProperty(self, propertyLabel,  value):
        self.m_properties["CalibrationParameters"][propertyLabel] = value

    def setStaticFilename(self,name):
        self.m_staticFilename=name

    def addChain(self,label,indexSegmentList):
        """
            Add a segment chain

            :Parameters:
                - `label` (str) - label of the chain
                - `indexSegmentList` (list of int) - index of segment

        """
        self.m_chains[label] = indexSegmentList

    def addJoint(self,label,proxLabel, distLabel, sequence):
        """
            Add a joint

            :Parameters:
                - `label` (str) - label of the chain
                - `proxLabel` (str) - label of the proximal segment
                - `distLabel` (str) - label of the distal segment
                - `sequence` (str) - sequence angle

        """

        j=Joint( label, proxLabel,distLabel,sequence)
        self.m_jointCollection.append(j)


    def addSegment(self,label,index,sideEnum, calibration_markers=[], tracking_markers=[],):
        """
            Add a segment

            :Parameters:
                - `label` (str) - label of the segment
                - `index` (str) - index of the segment
                - `sideEnum` (pyCGM2.enums) - body side
                - `lst_markerlabel` (list of str) - sequence angle
                - `tracking_markers` (list of str) - sequence angle

        """

        s=Segment(label,index,sideEnum,calibration_markers,tracking_markers)
        self.m_segmentCollection.append(s)


    def updateSegmentFromCopy(self,targetLabel, segmentToCopy):
        """
            Update a segment from a copy of  another segment instance

            :Parameters:
                - `targetLabel` (str) - label of the segment
                - `segmentToCopy` (pyCGM2.Model.CGM2.model.Segment) - pyCGM2.Model.CGM2.model.Segment instance

        """
        copiedSegment = copy.deepcopy(segmentToCopy)
        copiedSegment.name = targetLabel
        for i in range(0, len(self.m_segmentCollection)):
            if self.m_segmentCollection[i].name == targetLabel:
                self.m_segmentCollection[i] = copiedSegment


    def getSegment(self,label):
        """
            Get `Segment` from its label

            :Parameters:
                - `label` (str) - label of the Segment

            :Return:
                - `na` (pyCGM2.Model.CGM2.model.Segment) - pyCGM2.Model.CGM2.model.Segment instance


        """

        for it in self.m_segmentCollection:
            if it.name == label:
                return it

    def getSegmentIndex(self,label):
        """
            Get Segment index

            :Parameters:
                - `label` (str) - label of the Segment

            :Return:
                - `na` (int) - index


        """
        index=0
        for it in self.m_segmentCollection:
            if it.name == label:
                return index
            index+=1


    def getSegmentByIndex(self,index):
        """
            Get `Segment` from its index

            :Parameters:
                - `index` (int) - index of the Segment

            :Return:
                - `na` (pyCGM2.Model.CGM2.model.Segment) - pyCGM2.Model.CGM2.model.Segment instance


        """

        for it in self.m_segmentCollection:
            if it.index == index:
                return it



    def getJoint(self,label):
        """
            Get `Joint` from its label

            :Parameters:
                - `label` (str) - label of the joint

            :Return:
                - `na` (pyCGM2.Model.CGM2.model.Joint) - pyCGM2.Model.CGM2.model.Joint instance


        """

        for it in self.m_jointCollection:
            if it.m_label == label:
                return it

    def addAnthropoInputParameters(self,iDict,optional=None):
        """
            Add measured anthropometric data to the model

            :Parameters:
               - `iDict` (dict) - requried anthropometric data
               - `optionalMp` (dict) - optional anthropometric data

        """

        self.mp=iDict

        if optional is not None:
            self.mp.update(optional)






    def decomposeTrackingMarkers(self,acq,TechnicalFrameLabel):

    #        decompose marker and keep a noisy signal along one axis only.
    #
    #        :Parameters:
    #         - `acq` (btk-acq) -
    #         - `TechnicalFrameLabel` (str) - label of the technical Frame
    #
    #        .. todo::
    #
    #         - comment travailler avec des referentiels techniques differents par segment
    #         - rajouter des exceptions


        for seg in self.m_segmentCollection:

            for marker in seg.m_tracking_markers:

                nodeTraj= seg.getReferential(TechnicalFrameLabel).getNodeTrajectory(marker)
                markersTraj =acq.GetPoint(marker).GetValues()

                markerTrajectoryX=np.array( [ markersTraj[:,0], nodeTraj[:,1], nodeTraj[:,2]]).T
                markerTrajectoryY=np.array( [ nodeTraj[:,0], markersTraj[:,1], nodeTraj[:,2]]).T
                markerTrajectoryZ=np.array( [ nodeTraj[:,0], nodeTraj[:,1], markersTraj[:,2]]).T


                btkTools.smartAppendPoint(acq,marker+"-X",markerTrajectoryX,PointType=btk.btkPoint.Marker, desc="")
                btkTools.smartAppendPoint(acq,marker+"-Y",markerTrajectoryY,PointType=btk.btkPoint.Marker, desc="")
                btkTools.smartAppendPoint(acq,marker+"-Z",markerTrajectoryZ,PointType=btk.btkPoint.Marker, desc="")


    def displayStaticCoordinateSystem(self,aquiStatic,  segmentLabel, targetPointLabel, referential = "Anatomic" ):
        """
            Display a coordinate system. Its Axis are represented by 3 virtual markers suffixed by (_X,_Y,_Z)

            :Parameters:
                - `aquiStatic` (btkAcquisition) - btkAcquisition instance from a static c3d
                - `segmentLabel` (str) - segment label
                - `targetPointLabel` (str) - label of the point defining axis limits
                - `referential` (str) - type of segment coordinate system you want dislay ( if other than *Anatomic*, Technical Coordinate system will be displayed )

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

    def displayMotionCoordinateSystem(self,acqui,  segmentLabel, targetPointLabel, referential = "Anatomic" ):
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


    def displayMotionViconCoordinateSystem(self,acqui,  segmentLabel,targetPointLabelO,targetPointLabelX,targetPointLabelY,targetPointLabelZ, referential = "Anatomic" ):
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




# --------  MODEL COMPONANTS ---------
class Referential(object):
    """
         A `Referential` is a coordinate system
    """
    def __init__(self):
        self.static=frame.Frame()
        self.motion=[]
        self.relativeMatrixAnatomic = np.zeros((3,3))
        self.additionalInfos = dict()

    def setStaticFrame(self,Frame):
        """
            Set a `Frame` to the member Static of the `Referential`

            :Parameters:
                - `Frame` (pyCGM2.Model.CGM2.frame.Frame) - pyCGM2-Frame instance
        """
        self.static = Frame


    def addMotionFrame(self,Frame):
        """
             Add a `Frame` to the motion member of the `Referential`

            :Parameters:
                - `Frame` (pyCGM2.Model.CGM2.frame.Frame) - pyCGM2-Frame instance

        """
        self.motion.append(Frame)

    def getNodeTrajectory(self,label):
        """
            Get trajectory of a node

            :Parameters:
                - `label` (str) - label of the desired node

            :Return:
                - `pt` (numpy.array(:,3)) - values of the global point trajectory

        """

        node=self.static.getNode_byLabel(label)
        nFrames=len(self.motion)
        pt=np.zeros((nFrames,3))
        for i in range(0,nFrames):
            pt[i,:]=np.dot(self.motion[i].getRotation(),node.m_local)+ self.motion[i].getTranslation()

        return pt



class TechnicalReferential(Referential):
    """
        A `TechnicalReferential` inherit from `Referential`. This specification defines a *Technical* Coordinate system of the segment
    """
    def __init__(self, label):
        """
            :Parameters:
                - `label` (str) - label of the technical referential
        """
        #TODO - By labelling a technical referential, the code accepts multi-technical referentials per Segment. This feature should be removed


        super(TechnicalReferential, self).__init__()


        self.label=label


    def setRelativeMatrixAnatomic(self, array):
        """
            Set the relative rigid rotation of the anatomical Referential expressed in the technical referential (:math:`R^a_t`).

            :Parameters:
                - `array` (numpy.array(3,3) - rigid rotation
        """
        self.relativeMatrixAnatomic = array

class AnatomicalReferential(Referential):
    """
        A `AnatomicalReferential` inherit from `Referential`. This specification defines the *Anatomical* Coordinate system of a segment
    """
    def __init__(self):
        super(AnatomicalReferential, self).__init__()





class Segment(object):
    """
        A Segment defines a solid composing of the model
    """
    #TODO
    #       Referential doesn't distinguish anatomical frame  from technical frame
    #       used only for tracking gait and/or functional calibration
    #
    #       compute constant matrix rotation between each referential.static


    def __init__(self,label,index,sideEnum,calibration_markers=[], tracking_markers = []):
        """
            :Parameters:
                - `label` (str) - label of the segment
                - `index` (str) - index of the segment
                - `sideEnum` (pyCGM2.enums) - body side
                - `lst_markerlabel` (list of str) - sequence angle
                - `tracking_markers` (list of str) - sequence angle
        """

        self.name=label
        self.index=index
        self.side = sideEnum

        self.m_tracking_markers=tracking_markers
        self.m_calibration_markers = calibration_markers

        self.m_markerLabels=calibration_markers+tracking_markers
        self.referentials=[]
        self.anatomicalFrame =AnatomicalReferential()

        self.m_bsp = dict()
        self.m_bsp["mass"] = 0
        self.m_bsp["length"] = 0
        self.m_bsp["rog"] = 0
        self.m_bsp["com"] = np.zeros((3))
        self.m_bsp["inertia"] = np.zeros((3,3))

        self.m_externalDeviceWrenchs = list()
        self.m_externalDeviceBtkWrench = None
        self.m_proximalWrench = None

        self.m_proximalMomentContribution = dict()
        self.m_proximalMomentContribution["internal"] = None
        self.m_proximalMomentContribution["external"] = None
        self.m_proximalMomentContribution["inertia"] = None
        self.m_proximalMomentContribution["linearAcceleration"] = None
        self.m_proximalMomentContribution["gravity"] = None
        self.m_proximalMomentContribution["externalDevices"] = None
        self.m_proximalMomentContribution["distalSegments"] = None
        self.m_proximalMomentContribution["distalSegmentForces"] = None
        self.m_proximalMomentContribution["distalSegmentMoments"] = None

    def addTrackingMarkerLabel(self,label):
        """
            Add a tracking marker

            :Parameters:
                - `label` (str) - marker label
        """
        if label not in self.m_tracking_markers:
            self.m_tracking_markers.append(label)
            self.m_markerLabels.append(label)
        else:
            logging.debug("marker %s already in the tracking marker segment list" % label)

    def addCalibrationMarkerLabel(self,label):
        """
            Add a calibration marker

            :Parameters:
                - `label` (str) - marker label
        """

        if label not in self.m_calibration_markers:
            self.m_calibration_markers.append(label)
            self.m_markerLabels.append(label)
        else:
            logging.debug("marker %s already in the clibration marker segment list" % label)


    def resetMarkerLabels(self):
        self.m_markerLabels = list()
        self.m_markerLabels = self.m_tracking_markers + self.m_calibration_markers


    def addMarkerLabel(self,label):
        """
            Add a marker

            :Parameters:
                - `label` (str) - marker label
        """
        isFind=False
        i=0
        for marker in self.m_markerLabels:
            if label in marker:
                isFind=True
                index = i
            i+=1

        if isFind:
            logging.debug("marker %s already in the marker segment list" % label)


        else:
            self.m_markerLabels.append(label)

    def zeroingExternalDevice(self):
        """
            Zeroing external device wrench
        """
        self.m_externalDeviceWrenchs = list()
        self.m_externalDeviceBtkWrench = None

    def zeroingProximalWrench(self):
        """
            Zeroing proximal wrench
        """
        self.m_proximalWrench = None


    def downSampleExternalDeviceWrenchs(self,appf):
        """
            Downsample external device wrenchs

            :Parameters:
                - `appf` (int) - analog point per frame
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


    def isExternalDeviceWrenchsConnected(self):
        """
            Detect external device wrenchs
        """
        if self.m_externalDeviceWrenchs == []:
            return False
        else:
            return True

    def addExternalDeviceWrench(self, btkWrench):
        """
            Add an external device wrench

            :Parameters:
                - `btkWrench` (btk.wrench) - a btk wrench instance

        """
        self.m_externalDeviceWrenchs.append(btkWrench)



    def setMass(self, value):
        """
            Set segment mass

            :Parameters:
                - `value` (float) - mass

        """
        self.m_bsp["mass"] = value

    def setLength(self, value):
        """
            Set segment length

            :Parameters:
                - `value` (float) - length

        """
        self.m_bsp["length"] = value

    def setRog(self, value):
        """
            Set segment radius of giration

            :Parameters:
                - `value` (float) - radius of giration

        """
        self.m_bsp["rog"] = value

    def setComPosition(self, array3):
        """
            Set local position of the centre of mass

            :Parameters:
                - `array3` (numpy.array(3,)) - centre of mass position

        """
        self.m_bsp["com"] = array3

    def setInertiaTensor(self, array33):
        """
            Set segment inertia tensor

            :Parameters:
                - `array33` (numpy.array(3,3)) - tensor of inertia

        """
        self.m_bsp["inertia"] = array33










    def addTechnicalReferential(self,label):
        """
            Add a technical referential

            :Parameters:
                - `label` (str) - technical frame label
        """


        ref=TechnicalReferential(label)
        self.referentials.append(ref)



    def getReferential(self,label):
        """
            Access a referential from its label

            :Parameters:
                - `label` (str) - technical referential label

        """

        for tfIt in  self.referentials:
            if tfIt.label == label:
                return tfIt



    def getComTrajectory(self,exportBtkPoint=False, btkAcq=None):
        """
            Get global trajectory of the centre of mass

            :Parameters:
                - `exportBtkPoint` (bool) - enable export as btk point
                - `btkAcq` (btk acquisition) - a btk acquisition instance

            :Return:
                - `values` (numpy.array(n,3)) - values of the com trajectory
        """

        frameNumber = len(self.anatomicalFrame.motion)
        values = np.zeros((frameNumber,3))
        for i in range(0,frameNumber):
            values[i,:] = np.dot(self.anatomicalFrame.motion[i].getRotation() ,self.m_bsp["com"]) + self.anatomicalFrame.motion[i].getTranslation()

        if exportBtkPoint:
            if btkAcq != None:
                btkTools.smartAppendPoint(btkAcq,self.name + "_com",values,desc="com")
        return values


    def getComVelocity(self,pointFrequency,method = "spline"):
        """
            Get global linear velocity of the centre of mass

            :Parameters:
                - `pointFrequency` (double) - point frequency
                - `method` (str) - derivation method (spline, spline fitting)

            :Return:
                - `values` (numpy.array(n,3)) - values of the com velocity

            .. note :: if method doesnt recognize, numerical first order derivation is used

        """

        if method == "spline":
            values = derivation.splineDerivation(self.getComTrajectory(),pointFrequency,order=1)
        elif method == "spline fitting":
            values = derivation.splineFittingDerivation(self.getComTrajectory(),pointFrequency,order=1)
        else:
            values = derivation.firstOrderFiniteDifference(self.getComTrajectory(),pointFrequency)

        return values


    def getComAcceleration(self,pointFrequency,method = "spline", **options):
        """
            Get global linear acceleration of the centre of mass

            :Parameters:
                - `pointFrequency` (double) - point frequency
                - `method` (str) - derivation method (spline, spline fitting)
                - `**options` (kwargs) - options pass to method.

            :Return:
                - `values` (numpy.array(n,3)) - values of the com acceleration

            .. attention :: com trajectory can be smoothed  by calling fc and order, the cut-off frequency and the order of the low-pass filter respectively


            .. note ::

                if method doesnt recognize, numerical second order derivation is used


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





    def getAngularVelocity(self,sampleFrequency,method="conventional"):
        """
            Get angular velocity

            :Parameters:
                - `sampleFrequency` (double) - point frequency
                - `method` (str) - method used for computing the angular velocity

            :Return:
                - `AngularVelocValues` (numpy.array(n,3)) - values of the angular velocity

            .. note::

                The *pig* method reproduce a  bodybuilder code of the plug-in gait.

                .. raw:: python

                      AngVel = [ dot( NextPosR(:,2), PrevPosR(:,3) ); ...
                                 dot( NextPosR(:,3), PrevPosR(:,1) ); ...
                                 dot( NextPosR(:,1), PrevPosR(:,2) ); ...
                               ] ./ (2*SamplePeriod);

                The *conventional* method computes angular velocity through this statement :math:`\dot{R}R^t`

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


    def getAngularAcceleration(self,sampleFrequency):
        """
            Get angular acceleration

            :Parameters:
                - `sampleFrequency` (double) - point frequency

            :Return:
                - `values` (numpy.array(n,3)) - values of the angular accelration


            .. note:: A first order differention of the angular velocity is used
        """

        values = derivation.firstOrderFiniteDifference(self.getAngularVelocity(sampleFrequency),sampleFrequency)
        return values

class Joint(object):
    """
        a Joint defines relative motion between a proximal and a distal segment
    """

    def __init__(self, label, proxLabel,distLabel,sequence):
        """
            :Parameters:
                - `label` (str) - label of the chain
                - `proxLabel` (str) - label of the proximal segment
                - `distLabel` (str) - label of the distal segment
                - `sequence` (str) - sequence angle
        """
        self.m_label=label
        self.m_proximalLabel=proxLabel
        self.m_distalLabel=distLabel
        self.m_sequence=sequence
