# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 14:27:51 2015
@author: fleboeuf
"""

import numpy as np
import btk
import copy
import logging

import frame as cfr

from  pyCGM2.Tools import  btkTools
from pyCGM2.Math import  derivation

import pyCGM2.Signal.signal_processing as pyCGM2signal 



# -------- ABSTRACT MODEL ---------

class Model(object):
    """
    Model is a class-container collecting *segments* and *anthropometric data*.

            
    """

    def __init__(self):
        """ **Constructor**
        
        :Parameters:
         - `m_segmentCollection` (list) - collection of Segment
         - `mp` (dict) - anthropometric paramater
        """
        self.m_segmentCollection=[]
        self.m_jointCollection=[]
        self.mp=dict()
        self.mp_computed=dict()
        self.m_chains=dict()

    def addChain(self,label,indexSegmentList):
        self.m_chains[label] = indexSegmentList

    def addJoint(self,label,proxLabel, distLabel, sequence):
        """ add a joint 
        :Parameters:
        """
        j=Joint( label, proxLabel,distLabel,sequence)
        self.m_jointCollection.append(j)


    def addSegment(self,label,index,sideEnum, lst_markerlabel=[], tracking_markers=[],):
        """ add a Segment to the model
        
        :Parameters:
         - `label` (str) - label of the Segment ( Pelvis, Left Thigh...)
         - `lst_markerlabel` (list) - list of marker label

        """
        
        s=Segment(label,index,sideEnum,lst_markerlabel,tracking_markers)
        self.m_segmentCollection.append(s)


    def updateSegmentFromCopy(self,targetLabel, segmentToCopy):
        """ 
        update a segment by copying and renaming  another segment instance         
        
        """
        copiedSegment = copy.deepcopy(segmentToCopy)
        copiedSegment.name = targetLabel
        for i in range(0, len(self.m_segmentCollection)):
            if self.m_segmentCollection[i].name == targetLabel:
                self.m_segmentCollection[i] = copiedSegment




    def __repr__(self):
        return "Basis Model"




    def getSegment(self,label):
        """ get Segment from its label
        
        :Parameters:
         - `label` (str) - label of the Segment ( Pelvis, Left Thigh...)
                      
        :Returns Type: Segment 

        """        

        for it in self.m_segmentCollection:
            if it.name == label:
                return it

    def getSegmentIndex(self,label):
        """ 
        """        
        index=0
        for it in self.m_segmentCollection:
            if it.name == label:
                return index
            index+=1


    def getSegmentByIndex(self,index):
        """ get Segment by its index
        
        :Parameters:
                      
        :Returns Type: Segment 

        """        

        for it in self.m_segmentCollection:
            if it.index == index:
                return it



    def getJoint(self,label):
        """ get Segment from its label
        
        :Parameters:
        :Returns Type: Joint
        """        

        for it in self.m_jointCollection:
            if it.m_label == label:
                return it

    def addAnthropoInputParameter(self,iDict):
        """ add anthropometric data to the model  
        
        :Parameters:
         - `iDict` (dict) - dictionnary of anthropometric data
                      
        :Examples:
        ::

            mp={
           'mass'   : 75.0,         
           'height'   : 1760.0,         
           'pelvisDepth'   : 204.0,         
           'leftLegLength' : 820.0,
           'rightLegLength' : 820.0 ,
           'asisDistance' : 240.38 ,
           'leftKneeWidth' : 103.0,
           'rightKneeWidth' : 103.0,
           'leftAnkleWidth' : 71.0,
           'rightAnkleWidth' : 71.0 
            }  
            model.addAnthropoInputParameter(mp)            

        """

        self.mp=iDict


        
        
    def decomposeTrackingMarkers(self,acq,TechnicalFrameLabel):
        """
        decompose marker and keep a noisy signal along one axis only.

        :Parameters:
         - `acq` (btk-acq) - 
         - `TechnicalFrameLabel` (str) - label of the technical Frame
         
        .. todo:: 
        
         - comment travailler avec des referentiels techniques differents par segment
         - rajouter des exceptions
        """        
        
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



        

# --------  MODEL COMPONANTS ---------        

# --------  MODEL COMPONANTS ---------        
class Referential(object):
    def __init__(self):
        """ **Constructor**
        
        :Parameters:
         - `label` (str) - label of the Referential you want to build
        """ 
        self.static=cfr.Frame() 
        self.motion=[]
        self.relativeMatrixAnatomic = np.zeros((3,3))      # R ^a_t    

    def setStaticFrame(self,Frame):
        """ 
        
        :Parameters:
         - `Frame` (Frame) - Frame object appended 
        """
        self.static = Frame


    def addMotionFrame(self,Frame):
        """ add a Frame to the motion component fo the Referential
        
        :Parameters:
         - `Frame` (Frame) - Frame object appended 
        """
        self.motion.append(Frame)

    def getNodeTrajectory(self,label):
        """ call a Node and compyte its global trajectory
        
        :Parameters:
         - `label` (str) - label of the called node 
         - `display` (bool) - display print info 
        
        :Returns: a 3-column array with global trajectory
        :rtype: np.array((n,3))
        
        """        


        node=self.static.getNode_byLabel(label)
        nFrames=len(self.motion)        
        pt=np.zeros((nFrames,3))
        for i in range(0,nFrames):
            pt[i,:]=np.dot(self.motion[i].getRotation(),node.m_local)+ self.motion[i].getTranslation()            
        
        return pt 



class TechnicalReferential(Referential):
    def __init__(self, label):
        super(TechnicalReferential, self).__init__()
        self.label=label


    def setRelativeMatrixAnatomic(self, array):
        """ set the raltive matrix anatomic in technic( static). 
        
        it's the matrix R(^a_t) 
        
        :Parameters:
         - `array` (np.array(3,3)) - matrix  
       
        """
        self.relativeMatrixAnatomic = array

class AnatomicalReferential(Referential):
    def __init__(self):
        super(AnatomicalReferential, self).__init__()





class Segment(object):
    """ 
    .. todo:: 
       
       Referential doesn't distinguish anatomical frame  from technical frame
       used only for tracking gait and/or functional calibration
 
       compute constant matrix rotation between each referential.static      

    """
    def __init__(self,label,index,sideEnum ,lst_markerlabel=[], tracking_markers = []):
        """ **Constructor**

        :Parameters:
         - `label` (str) - label of the segment you want to create  
         - `lst_markerlabel` (lst) - list of marker label
        """

        self.name=label
        self.index=index
        self.side = sideEnum
        self.m_markerLabels=lst_markerlabel
        self.m_tracking_markers=tracking_markers
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


    def zeroingExternalDevice(self):
        self.m_externalDeviceWrenchs = list()
        self.m_externalDeviceBtkWrench = None

    def zeroingProximalWrench(self):
        self.m_proximalWrench = None


    def downSampleExternalDeviceWrenchs(self,appf):
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
        if self.m_externalDeviceWrenchs == []:
            return False
        else:
            return True

    def addExternalDeviceWrench(self, btkWrench):
        self.m_externalDeviceWrenchs.append(btkWrench)

   

    def setMass(self, value):
        self.m_bsp["mass"] = value        

    def setLength(self, value):
        self.m_bsp["length"] = value        

    def setRog(self, value):
        self.m_bsp["rog"] = value        

    def setComPosition(self, array3):
        self.m_bsp["com"] = array3        

    def setInertiaTensor(self, array33):
        self.m_bsp["inertia"] = array33        


    def addMarkerLabel(self,label):
        """ add a marker to the list

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
            logging.warning("marker %s already in the marker segment list" % label)


        else:
            self.m_markerLabels.append(label)


    def addTrackingMarkerLabel(self,label):
        """ add a marker to the list of tracking markers

        :Parameters:
         - `label` (str) - marker label  
        """
        isFind=False
        i=0
        for marker in self.m_tracking_markers:
            if label in marker:
                isFind=True
                index = i
            i+=1

        if isFind:
            logging.warning("marker %s already in the tracking marker segment list" % label) 

        else:
            self.m_tracking_markers.append(label)




    def addTechnicalReferential(self,label):
        """ add a referential by its label
        :Parameters:
         - `label` (str) - technical frame label  
        """

  
        ref=TechnicalReferential(label)
        self.referentials.append(ref)



    def getReferential(self,label):
        """ get a referential by its label
        :Parameters:
         - `label` (str) - referential label  

        """ 

        for tfIt in  self.referentials:
            if tfIt.label == label:
                return tfIt
                

    
        #self.m_angularVelocity = AngularVelocValues    

        
    

    def getComTrajectory(self,exportBtkPoint=False, btkAcq=None):

        frameNumber = len(self.anatomicalFrame.motion)         
        values = np.zeros((frameNumber,3))   
        for i in range(0,frameNumber):    
            values[i,:] = np.dot(self.anatomicalFrame.motion[i].getRotation() ,self.m_bsp["com"]) + self.anatomicalFrame.motion[i].getTranslation() 

        if exportBtkPoint:
            if btkAcq != None:
                btkTools.smartAppendPoint(btkAcq,self.name + "_com",values,desc="com")
        return values


    def getComVelocity(self,pointFrequency,method = "spline"):
        if method == "spline":
            values = derivation.splineDerivation(self.getComTrajectory(),pointFrequency,order=1) 
        elif method == "spline fitting":
            values = derivation.splineFittingDerivation(self.getComTrajectory(),pointFrequency,order=1) 
        else:
            values = derivation.firstOrderFiniteDifference(self.getComTrajectory(),pointFrequency)
        
        return values


    def getComAcceleration(self,pointFrequency,method = "spline", **options):
        
        valueCom = self.getComTrajectory()
        if "fc" in options.keys() and  "order" in options.keys():        
            valueCom = pyCGM2signal.arrayLowPassFiltering(valueCom,pointFrequency,options["order"],options["fc"]  )

        if method == "spline":
            values = derivation.splineDerivation(valueCom,pointFrequency,order=2) 
        elif method == "spline fitting":
            values = derivation.splineFittingDerivation(self.getComTrajectory(),pointFrequency,order=2) 
        else:
            values = derivation.secondOrderFiniteDifference(valueCom,pointFrequency)
        
        return values

         



    def getAngularVelocity(self,sampleFrequency,method="conventional"): 
        """ short description
        :Parameters:
         - `?` (?) - desc  

          AngVel = [ ...
              dot( NextPosR(:,2), PrevPosR(:,3) ); ...
              dot( NextPosR(:,3), PrevPosR(:,1) ); ...
              dot( NextPosR(:,1), PrevPosR(:,2) ); ...
              ] ./ (2*SamplePeriod);

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
        
        values = derivation.firstOrderFiniteDifference(self.getAngularVelocity(sampleFrequency),sampleFrequency)
        return values
             
class Joint(object):
    """

    """

    def __init__(self, label, proxLabel,distLabel,sequence):
        """ **Constructor**
        :Parameters:
        """
        self.m_label=label
        self.m_proximalLabel=proxLabel
        self.m_distalLabel=distLabel
        self.m_sequence=sequence
    
