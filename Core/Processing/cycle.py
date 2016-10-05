# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 10:52:09 2016

@author: fabien Leboeuf ( Salford Univ)
"""

import numpy as np
import pdb


import ma.io
import ma.body



import pyCGM2.Core.Tools.trialTools as CGM2trialTools
import pyCGM2.Core.Math.normalisation  as MathNormalisation



class Cycle(ma.Node):
    """
    pour definir un label, il faut passer par la methode setName de l objet node

    """    
    
    
    def __init__(self,trial,startTime,endTime,context, enableFlag = True):
        """
        """
  
        nodeLabel = "Cycle"   
        super(Cycle,self).__init__(nodeLabel)
        self.trial=trial
        self.pointfrequency = trial.findChild(ma.T_TimeSequence,"",[["type",ma.TimeSequence.Type_Marker]]).sampleRate()
        self.analogfrequency = trial.findChild(ma.T_TimeSequence,"",[["type",ma.TimeSequence.Type_Analog]]).sampleRate() 
        self.appf =  self.analogfrequency / self.pointfrequency
        self.firstFrame = trial.findChild(ma.T_TimeSequence,"",[["type",ma.TimeSequence.Type_Marker]]).startTime() * self.pointfrequency


        self.begin =  round(startTime * self.pointfrequency) + 1
        self.end = round(endTime * self.pointfrequency) + 1
        self.context=context
        self.enableFlag = enableFlag
                

        

        
        self.discreteDataList=[]

        print "cycle makes from Frame %d   to  %d   (%s) " % (self.begin, self.end, self.context)
        


    def setEnableFlag(self,flag):
        self.enableFlag = flag        
        
    def addDiscreteData(self,label,value,instant):
        pass #todo

    def getPointTimeSequenceData(self,markerLabel):
        if CGM2trialTools.isTimeSequenceExist(self.trial,markerLabel):
            return  self.trial.findChild(ma.T_TimeSequence, markerLabel).data()[self.begin-self.firstFrame:self.end-self.firstFrame+1,0:3] # 0.3 because openma::Ts includes a forth column (i.e residual)  
        else:
            raise Exception("[pyCGM2] marker %s doesn t exist"% markerLabel )


    def getPointTimeSequenceDataNormalized(self,markerLabel):
        data = self.getPointTimeSequenceData(markerLabel)
        return  MathNormalisation.timeSequenceNormalisation(101,data)

    def getAnalogTimeSequenceData(self,analogLabel):
        if CGM2trialTools.isTimeSequenceExist(self.trial,analogLabel):
            return  self.trial.findChild(ma.T_TimeSequence, analogLabel).data()[(self.begin-self.firstFrame) * self.appf : (self.end-self.firstFrame+1) * self.appf,:]
        else:
            raise Exception("[pyCGM2] Analog %s doesn t exist"% analogLabel )
        
    def getAnalogTimeSequenceDataNormalized(self,analogLabel):
        data = self.getAnalogTimeSequenceData(analogLabel)
        return  MathNormalisation.timeSequenceNormalisation(101,data)        

    def getEvents(self,context="All"):
        events = ma.Node("Events")
        evsn = self.trial.findChild(ma.T_Node,"SortedEvents")
        for ev in evsn.findChildren(ma.T_Event):
            if context==("Left" or "Right") :
                if ev.context() ==context:
                    if round(ev.time() * self.pointfrequency) + 1  >self.begin and  round(ev.time() * self.pointfrequency) + 1<self.end:
                        ev.addParent(events)
            else:
                if round(ev.time() * self.pointfrequency) + 1 > self.begin and  round(ev.time() * self.pointfrequency) + 1 < self.end:
                    ev.addParent(events)
        return events
        
        
class GaitCycle(Cycle):

    STP_LABELS=["duration","cadence", "stanceDuration",  "stancePhase", 
                "swingDuration", "swingPhase", "doubleStance1", "doubleStance2",
                "simpleStance", "strideLength", "stepLength",
                "strideWidth", "speed"]

    
    def __init__(self,gaitTrial,startTime,endTime,context, enableFlag = True,
                 longitudinal_axis=0,lateral_axis=1):
        super(GaitCycle, self).__init__(gaitTrial,startTime,endTime,context, enableFlag = enableFlag)
        
        #ajout des oppositeFO, contraFO,oopositeFS 
        evs=self.getEvents()
        if context=="Right":
            oppositeSide="Left"
        elif context=="Left":
            oppositeSide="Right"
        for ev in evs.findChildren(ma.T_Event):
            if ev.name() == "Foot Off" and ev.context()==oppositeSide:
                oppositeFO= round(ev.time() * self.pointfrequency) + 1 
            if ev.name() == "Foot Strike" and ev.context()==oppositeSide:
                oppositeFS= round(ev.time() * self.pointfrequency) + 1
            if ev.name() == "Foot Off" and ev.context()==context:
                contraFO = round(ev.time() * self.pointfrequency) + 1
        if oppositeFO > oppositeFS:
            raise Exception("[pyCGM2] : check your c3d - Gait event error")
        
        
        self.m_oppositeFO=oppositeFO
        self.m_oppositeFS=oppositeFS
        self.m_contraFO=contraFO
        self.m_normalizedOppositeFO=round(np.divide(float(self.m_oppositeFO - self.begin),float(self.end-self.begin))*100)
        self.m_normalizedOppositeFS=round(np.divide(float(self.m_oppositeFS - self.begin),float(self.end-self.begin))*100)
        self.m_normalizedContraFO=round(np.divide(float(self.m_contraFO - self.begin),float(self.end-self.begin))*100)
        
        
        self.__computeSpatioTemporalParameter(longitudinal_axis,lateral_axis)        
        
        
    def __computeSpatioTemporalParameter(self,longitudinal_axis,lateral_axis):

        duration = np.divide((self.end-self.begin),self.pointfrequency)

        pst = ma.Node("stp",self)
        pst.setProperty("duration", duration)
        pst.setProperty("cadence", np.divide(60.0,duration))
        stanceDuration=np.divide(np.abs(self.m_contraFO - self.begin) , self.pointfrequency)
        pst.setProperty("stanceDuration", stanceDuration)
        pst.setProperty("stancePhase", round(np.divide(stanceDuration,duration)*100))        
        swingDuration=np.divide(np.abs(self.m_contraFO - self.end) , self.pointfrequency)        
        pst.setProperty("swingDuration", swingDuration)
        pst.setProperty("swingPhase", round(np.divide(swingDuration,duration)*100 ))
        pst.setProperty("doubleStance1", round(np.divide(np.divide(np.abs(self.m_oppositeFO - self.begin) , self.pointfrequency),duration)*100))
        pst.setProperty("doubleStance2", round(np.divide(np.divide(np.abs(self.m_contraFO - self.m_oppositeFS) , self.pointfrequency),duration)*100))
        pst.setProperty("simpleStance ",round(np.divide(np.divide(np.abs(self.m_oppositeFO - self.m_oppositeFS) , self.pointfrequency),duration)*100))

        if self.context == "Left":

            if CGM2trialTools.isTimeSequenceExist(self.trial,"LHEE") and CGM2trialTools.isTimeSequenceExist(self.trial,"RHEE") and CGM2trialTools.isTimeSequenceExist(self.trial,"LTOE"):
                
                strideLength=np.abs(self.getPointTimeSequenceData("LHEE")[self.end-self.begin,longitudinal_axis] -\
                                    self.getPointTimeSequenceData("LHEE")[0,longitudinal_axis])/1000.0
                pst.setProperty("strideLength", strideLength)

                stepLength = np.abs(self.getPointTimeSequenceData("RHEE")[self.m_oppositeFS-self.begin,longitudinal_axis] -\
                                    self.getPointTimeSequenceData("LHEE")[0,longitudinal_axis])/1000.0
                pst.setProperty("stepLength", stepLength)
                
                strideWidth = np.abs(self.getPointTimeSequenceData("LTOE")[self.end-self.begin,lateral_axis] -\
                                     self.getPointTimeSequenceData("RHEE")[0,lateral_axis])/1000.0
                pst.setProperty("strideWidth", strideWidth)
                
                pst.setProperty("speed",np.divide(strideLength,duration)) 
                
                
        if self.context == "Right":

            if CGM2trialTools.isTimeSequenceExist(self.trial,"RHEE") and CGM2trialTools.isTimeSequenceExist(self.trial,"LHEE") and CGM2trialTools.isTimeSequenceExist(self.trial,"RTOE"):
                
                strideLength=np.abs(self.getPointTimeSequenceData("RHEE")[self.end-self.begin,longitudinal_axis] -\
                                    self.getPointTimeSequenceData("RHEE")[0,longitudinal_axis])/1000.0
                pst.setProperty("strideLength", strideLength)

                stepLength = np.abs(self.getPointTimeSequenceData("RHEE")[self.m_oppositeFS-self.begin,longitudinal_axis] -\
                                    self.getPointTimeSequenceData("LHEE")[0,longitudinal_axis])/1000.0
                pst.setProperty("stepLength", stepLength)
                
                strideWidth = np.abs(self.getPointTimeSequenceData("RTOE")[self.end-self.begin,lateral_axis] -\
                                     self.getPointTimeSequenceData("LHEE")[0,lateral_axis])/1000.0
                pst.setProperty("strideWidth", strideWidth)
                
                pst.setProperty("speed",np.divide(strideLength,duration))

    def getSpatioTemporalParameter(self,label):
        stpNode = self.findChild(ma.T_Node,"stp")
        return stpNode.property(label).cast()

# ----- FILTER -----
class CyclesFilter: 
    """
    get the different element of cyclesAnalysis and construct the object cyclesAnalysis
    """
 
    __builder = None
 
    def setBuilder(self, builder):
        self.__builder = builder
 

    def build(self):
        print "####### CYCLES FILTER ########"
        cycles = Cycles()
 
        print " ----- build spatio-Temporal cycles -----" 
        spatioTemporalElements = self.__builder.getSpatioTemporal()
        cycles.setSpatioTemporalCycles(spatioTemporalElements)        

        print " ----- build Kinematics cycles -----"
        kinematicElements = self.__builder.getKinematics()            
        cycles.setKinematicCycles(kinematicElements)
  
        print " ----- build Kinetic cycles -----"
        kineticElements = self.__builder.getKinetics()
        cycles.setKineticCycles(kineticElements)

        print " ----- build emg cycles -----"
        emgElements = self.__builder.getEmg()
        cycles.setEmgCycles(emgElements)

        return cycles   




# ----- BUILDER PATTERN -----

# --- object to build 
class Cycles(): 
    """
    """
   
    def __init__(self):
        self.spatioTemporalCycles = None
        self.kinematicCycles = None
        self.kineticCycles = None
        self.emgCycles = None
        

    def setSpatioTemporalCycles(self,spatioTemporalCycles_instance):   
        self.spatioTemporalCycles = spatioTemporalCycles_instance    
       
    def setKinematicCycles(self,kinematicCycles_instance):   
        self.kinematicCycles = kinematicCycles_instance

    def setKineticCycles(self,kineticCycles_instance):   
        self.kineticCycles = kineticCycles_instance

    def setEmgCycles(self,emgCycles_instance):   
        self.emgCycles = emgCycles_instance    
    
    
# --- builders 
class CyclesBuilder(object):
    """
    Abstract Builder
    """
    def __init__(self,spatioTemporalTrials=None,kinematicTrials=None,kineticTrials=None,emgTrials=None,longitudinal_axis=0,lateral_axis=1):
        self.spatioTemporalTrials =spatioTemporalTrials        
        self.kinematicTrials =kinematicTrials
        self.kineticTrials =kineticTrials
        self.emgTrials =emgTrials
        self.longitudinal_axis=longitudinal_axis
        self.lateral_axis=lateral_axis


    def getSpatioTemporal(self): pass   
    def getKinematics(self):pass
    def getKinetics(self):pass
    def getEmg(self):pass

class GaitCyclesBuilder(CyclesBuilder):
    """
    CONCRETE builder for gait analysis
    """

    def __init__(self,spatioTemporalTrials=None,kinematicTrials=None,kineticTrials=None,emgTrials=None,longitudinal_axis=0,lateral_axis=1):
        super(GaitCyclesBuilder, self).__init__(
            spatioTemporalTrials = spatioTemporalTrials,
            kinematicTrials=kinematicTrials,
            kineticTrials = kineticTrials,
            emgTrials = emgTrials,
            longitudinal_axis=longitudinal_axis,
            lateral_axis=lateral_axis
            )

           
    def getSpatioTemporal(self):
        
        if self.spatioTemporalTrials is not None:
            spatioTemporalCycles=list()
            for trial in  self.spatioTemporalTrials:
                 
                context = "Left"
                left_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                    left_fs_times.append(ev.time())

                for i in range(0, len(left_fs_times)-1):
                    spatioTemporalCycles.append (GaitCycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                   context,
                                                   longitudinal_axis = self.longitudinal_axis,
                                                   lateral_axis =self.lateral_axis)) 

                context = "Right"
                right_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                    right_fs_times.append(ev.time())

                for i in range(0, len(right_fs_times)-1):
                    spatioTemporalCycles.append (GaitCycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                   context,
                                                   longitudinal_axis = self.longitudinal_axis,
                                                   lateral_axis =self.lateral_axis))

            return spatioTemporalCycles
        else:
            return None

    def getKinematics(self):
        
        if self.kinematicTrials is not None:
            kinematicCycles=list()
            for trial in  self.kinematicTrials:
                 
                context = "Left"
                left_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                    left_fs_times.append(ev.time())

                for i in range(0, len(left_fs_times)-1):
                    kinematicCycles.append (GaitCycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                   context,
                                                   longitudinal_axis = self.longitudinal_axis,
                                                   lateral_axis =self.lateral_axis)) 

                context = "Right"
                right_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                    right_fs_times.append(ev.time())

                for i in range(0, len(right_fs_times)-1):
                    kinematicCycles.append (GaitCycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                   context,
                                                   longitudinal_axis = self.longitudinal_axis,
                                                   lateral_axis =self.lateral_axis))

            return kinematicCycles
        else:
            return None
            
    def getKinetics(self):
        
        if self.kineticTrials is not None:
            kineticCycles=list()
            for trial in  self.kineticTrials:
                 
                context = "Left"
                left_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                    left_fs_times.append(ev.time())

                for i in range(0, len(left_fs_times)-1):
                    kineticCycles.append (GaitCycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                   context,
                                                   longitudinal_axis = self.longitudinal_axis,
                                                   lateral_axis =self.lateral_axis)) 

                context = "Right"
                right_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                    right_fs_times.append(ev.time())

                for i in range(0, len(right_fs_times)-1):
                    kineticCycles.append (GaitCycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                   context,
                                                   longitudinal_axis = self.longitudinal_axis,
                                                   lateral_axis =self.lateral_axis))

            return kineticCycles
        else:
            return None
            
    def getEmg(self):
        
        if self.emgTrials is not None:
            emgCycles=list()
            for trial in  self.emgTrials:
                 
                context = "Left"
                left_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                    left_fs_times.append(ev.time())

                for i in range(0, len(left_fs_times)-1):
                    emgCycles.append (GaitCycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                   context,
                                                   longitudinal_axis = self.longitudinal_axis,
                                                   lateral_axis =self.lateral_axis)) 

                context = "Right"
                right_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                    right_fs_times.append(ev.time())

                for i in range(0, len(right_fs_times)-1):
                    emgCycles.append (GaitCycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                   context,
                                                   longitudinal_axis = self.longitudinal_axis,
                                                   lateral_axis =self.lateral_axis))

            return emgCycles
        else:
            return None