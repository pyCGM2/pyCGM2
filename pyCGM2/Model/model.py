# -*- coding: utf-8 -*-
import numpy as np
import copy
import logging

try: 
    from pyCGM2 import btk
except:
    logging.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk

from pyCGM2.Model import frame


from pyCGM2 import enums
from pyCGM2.Model import motion
from  pyCGM2.Tools import  btkTools
from pyCGM2.Math import  derivation
from  pyCGM2.Signal import signal_processing


class ClinicalDescriptor(object):
    def __init__(self,dataType,jointOrSegmentLabel, indexes,coefficients, offsets,**options):
        self.type = dataType
        self.label = jointOrSegmentLabel
        self.infos = dict()
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
        self.m_clinicalDescriptors= []
        self.m_csDefinitions = []
        self.m_bodypart=None
        self.m_centreOfMass=None

    def __repr__(self):
        return "Basis Model"

    def setBodyPart(self, bodypart):
        self.m_bodypart = bodypart

    def getBodyPart(self):
        return self.m_bodypart


    def setCentreOfMass(self,com):
        self.m_centreOfMass = com
    def getCentreOfMass(self):
        return self.m_centreOfMass


    def setProperty(self, propertyLabel,  value):
        self.m_properties[propertyLabel] = value

    def getProperty(self, propertyLabel):
        try:
            return self.m_properties[propertyLabel]
        except:
            raise ("property Label doesn t find")

    def setCalibrationProperty(self, propertyLabel,  value):
        self.m_properties["CalibrationParameters"][propertyLabel] = value

    def isProperty(self,label):
        return True if label in self.m_properties.keys() else False

    def isCalibrationProperty(self,label):
        return True if label in self.m_properties["CalibrationParameters"].keys() else False

    def checkCalibrationProperty(self,CalibrationParameterLabel,value):
        if self.isCalibrationProperty(CalibrationParameterLabel):
            if self.m_properties["CalibrationParameters"][CalibrationParameterLabel] == value:
                return True
            else:
                return False
        else:
            return False
            logging.warning("[pyCGM2] : CalibrationParameterLabel doesn t exist")

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

    def addJoint(self,label,proxLabel, distLabel, sequence, nodeLabel):
        """
            Add a joint

            :Parameters:
                - `label` (str) - label of the chain
                - `proxLabel` (str) - label of the proximal segment
                - `distLabel` (str) - label of the distal segment
                - `sequence` (str) - sequence angle

        """

        j=Joint( label, proxLabel,distLabel,sequence,nodeLabel)
        self.m_jointCollection.append(j)


    def addSegment(self,label,index,sideEnum, calibration_markers=[], tracking_markers=[],cloneOf=False):
        """
            Add a segment

            :Parameters:
                - `label` (str) - label of the segment
                - `index` (str) - index of the segment
                - `sideEnum` (pyCGM2.enums) - body side
                - `lst_markerlabel` (list of str) - sequence angle
                - `tracking_markers` (list of str) - sequence angle

        """

        seg=Segment(label,index,sideEnum,calibration_markers,tracking_markers)
        if cloneOf is True:
            seg.m_isCloneOf = True
        self.m_segmentCollection.append(seg)


    def updateSegmentFromCopy(self,targetLabel, segmentToCopy):
        """
            Update a segment from a copy of  another segment instance

            :Parameters:
                - `targetLabel` (str) - label of the segment
                - `segmentToCopy` (pyCGM2.Model.CGM2.model.Segment) - pyCGM2.Model.CGM2.model.Segment instance

        """
        copiedSegment = copy.deepcopy(segmentToCopy)
        copiedSegment.name = targetLabel

        isClone = True if self.getSegment(targetLabel).m_isCloneOf else False
        for i in range(0, len(self.m_segmentCollection)):
            if self.m_segmentCollection[i].name == targetLabel:
                self.m_segmentCollection[i] = copiedSegment
                self.m_segmentCollection[i].m_isCloneOf = isClone


    def removeSegment(self,segmentlabels):
        """
            remove `Segment` from its label

            :Parameters:
                - `label` (list) - segment labels to remove

            :Return:
                - `na` (pyCGM2.Model.CGM2.model.Segment) - pyCGM2.Model.CGM2.model.Segment instance


        """
        segment_list = [it for it in self.m_segmentCollection if it.name not in segmentlabels]
        self.m_segmentCollection = segment_list

    def removeJoint(self,jointlabels):
        """
            remove `joint` from its label

            :Parameters:
                - `jointlabels` (list) - joint labels to remove

            :Return:
                - `na` (pyCGM2.Model.CGM2.model.Segment) - pyCGM2.Model.CGM2.model.Segment instance


        """
        joint_list = [it for it in self.m_jointCollection if it.m_label not in jointlabels]
        self.m_jointCollection = joint_list



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

    def getSegmentList(self):
        """
            Get list of `Segment` labels

            :Parameters:


            :Return:
                - `labels` (list) - list of segment label


        """
        return [it.name for it in self.m_segmentCollection]


    def getJointList(self):
        """
            Get list of `Joint` labels

            :Parameters:


            :Return:
                - `labels` (list) - list of joint label


        """
        return [it.m_label for it in self.m_jointCollection]



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

    def setClinicalDescriptor(self,jointOrSegmentLabel,dataType, indexes,coefficients, offsets,**options):


        descriptor = ClinicalDescriptor(dataType, jointOrSegmentLabel, indexes,coefficients, offsets,**options)
        self.m_clinicalDescriptors.append(descriptor)

    def getClinicalDescriptor(self,dataType,jointOrSegmentLabel,projection=None):

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
            logging.debug("[pyCGM2] : descriptor [ type: %s - label: %s]  not found" %(dataType.name,jointOrSegmentLabel))

        return infos

    def setCoordinateSystemDefinition(self,segmentLabel,coordinateSystemLabel, referentialType):
        dic = {"segmentLabel": segmentLabel,"coordinateSystemLabel": coordinateSystemLabel,"referentialType": referentialType}
        self.m_csDefinitions.append(dic)


class Model6Dof(Model):
    """

    """
    def __init__(self):
        super(Model6Dof, self).__init__()

    def _calibrateTechnicalSegment(self,aquiStatic, segName, dictRef,frameInit,frameEnd, options=None):

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

    def _calibrateAnatomicalSegment(self,aquiStatic, segName, dictAnatomic,frameInit,frameEnd, options=None):

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


    def computeMotionTechnicalFrame(self,aqui,segName,dictRef,method,options=None):
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

    def computeMotionAnatomicalFrame(self,aqui,segName,dictAnatomic,options=None):

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
        self.relativeMatrixAnatomic=np.eye(3,3)


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

        self.m_info = dict()
        self.m_isCloneOf = False

    def removeTrackingMarker(self,label):
        """
            Add a tracking marker

            :Parameters:
                - `label` (str) - marker label
        """
        if label in self.m_tracking_markers:
            self.m_tracking_markers.remove(label)
            self.m_markerLabels.remove(label)
        else:
            logging.debug("tracking marker %s  remove" % label)


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

    def __init__(self, label, proxLabel,distLabel,sequence,nodeLabel):
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
        self.m_nodeLabel=nodeLabel
