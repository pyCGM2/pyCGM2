# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 10:52:09 2016

@author: fabien Leboeuf ( Salford Univ)
"""

import numpy as np
import pdb
import logging

import ma.io
import ma.body



import pyCGM2.Core.Tools.trialTools as CGM2trialTools
import pyCGM2.Core.Math.normalisation  as MathNormalisation

#---- MODULE METHODS ------

def spatioTemporelParameter_descriptiveStats(cycles,label,context):
    
    """   
    """

    outDict=dict()    
    
    n=len([cycle for cycle in cycles if cycle.enableFlag and cycle.context==context]) # list comprehension , get number of enabled cycle 
    val = np.zeros((n))
    
    i=0
    for cycle in cycles:
        if cycle.enableFlag and cycle.context==context:
            val[i] = cycle.getSpatioTemporalParameter(label)
            i+=1
    outDict = {'mean':np.mean(val),'std':np.std(val),'median':np.median(val),'values': val}
            
    return outDict


def point_descriptiveStats(cycles,label,context):
    """
    """
       
    outDict=dict()    
    
    n=len([cycle for cycle in cycles if cycle.enableFlag and cycle.context==context]) # list comprehension , get number of enabled cycle 
    
    x=np.empty((101,n))
    y=np.empty((101,n))
    z=np.empty((101,n))

    listOfPointValues=list()

    i=0
    for cycle in cycles:
        if cycle.enableFlag and cycle.context==context:
            tmp = cycle.getPointTimeSequenceDataNormalized(label)
            x[:,i]=tmp[:,0]
            y[:,i]=tmp[:,1]
            z[:,i]=tmp[:,2]
            
            listOfPointValues.append(tmp)
            
            i+=1
                
            
    meanData=np.array(np.zeros((101,3)))    
    meanData[:,0]=np.mean(x,axis=1)
    meanData[:,1]=np.mean(y,axis=1)
    meanData[:,2]=np.mean(z,axis=1)
    
    stdData=np.array(np.zeros((101,3)))    
    stdData[:,0]=np.std(x,axis=1)
    stdData[:,1]=np.std(y,axis=1)
    stdData[:,2]=np.std(z,axis=1)


    medianData=np.array(np.zeros((101,3)))    
    medianData[:,0]=np.median(x,axis=1)
    medianData[:,1]=np.median(y,axis=1)
    medianData[:,2]=np.median(z,axis=1)


    outDict = {'mean':meanData, 'median':medianData, 'std':stdData, 'values': listOfPointValues }
    

            
    return outDict
    
    
    
def analog_descriptiveStats(cycles,label,context):

    outDict=dict()    


    
    n=len([cycle for cycle in cycles if cycle.enableFlag and cycle.context==context]) # list comprehension , get number of enabled cycle 
    
    x=np.empty((1001,n))

    i=0
    for cycle in cycles:
        if cycle.enableFlag and cycle.context==context:
            tmp = cycle.getAnalogTimeSequenceDataNormalized(label)
            x[:,i]=tmp[:,0]
          
            
            i+=1

    x_resize=x[0:1001:10,:]
            
    meanData=np.array(np.zeros((101,1)))    
    meanData[:,0]=np.mean(x_resize,axis=1)
    
    stdData=np.array(np.zeros((101,1)))    
    stdData[:,0]=np.std(x_resize,axis=1)

    medianData=np.array(np.zeros((101,1)))    
    medianData[:,0]=np.median(x_resize,axis=1)


    outDict = {'mean':meanData, 'median':medianData, 'std':stdData, 'values': x_resize }
            
    return outDict

#---- CLASSES ------

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

        logging.info("cycle makes from Frame %d   to  %d   (%s) " % (self.begin, self.end, self.context))
        


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
        pst.setProperty("simpleStance", round(np.divide(np.divide(np.abs(self.m_oppositeFO - self.m_oppositeFS) , self.pointfrequency),duration)*100))

        #pst.setProperty("simpleStance3 ",15.0 )

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
                try: 
                    strideLength=np.abs(self.getPointTimeSequenceData("RHEE")[self.end-self.begin,longitudinal_axis] -\
                                        self.getPointTimeSequenceData("RHEE")[0,longitudinal_axis])/1000.0
                     
                    strideWidth = np.abs(self.getPointTimeSequenceData("RTOE")[self.end-self.begin,lateral_axis] -\
                                     self.getPointTimeSequenceData("LHEE")[0,lateral_axis])/1000.0 
                                        
                except IndexError:
                    strideLength=np.abs(self.getPointTimeSequenceData("RHEE")[(self.end-self.begin)-1,longitudinal_axis] -\
                                        self.getPointTimeSequenceData("RHEE")[0,longitudinal_axis])/1000.0
                                        
                    strideWidth = np.abs(self.getPointTimeSequenceData("RTOE")[self.end-self.begin-1,lateral_axis] -\
                                     self.getPointTimeSequenceData("LHEE")[0,lateral_axis])/1000.0                                         
                                        
                    logging.error("The last frame of the c3d is probably a foot strike. PyCGM2 takes the before-end Frame fro computing bth stride length and stride width" )

                pst.setProperty("strideLength", strideLength)
                pst.setProperty("strideWidth", strideWidth)

                stepLength = np.abs(self.getPointTimeSequenceData("RHEE")[self.m_oppositeFS-self.begin,longitudinal_axis] -\
                                    self.getPointTimeSequenceData("LHEE")[0,longitudinal_axis])/1000.0
                pst.setProperty("stepLength", stepLength)
                
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
        cycles = Cycles()
 
        spatioTemporalElements = self.__builder.getSpatioTemporal()
        cycles.setSpatioTemporalCycles(spatioTemporalElements)        

        kinematicElements = self.__builder.getKinematics()            
        cycles.setKinematicCycles(kinematicElements)
  
        kineticElements = self.__builder.getKinetics()
        cycles.setKineticCycles(kineticElements)

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
                
                flag_kinetics,times = CGM2trialTools.isKineticFlag(trial)
                
                if flag_kinetics:
                    context = "Left"
                    left_fs_times=list()
                    for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                        left_fs_times.append(ev.time())
                    
                    logging.info("--left kinetic cycle--")
                    for i in range(0, len(left_fs_times)-1):
                        if left_fs_times[i] in times:
                            kineticCycles.append (GaitCycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                           context,
                                                           longitudinal_axis = self.longitudinal_axis,
                                                           lateral_axis =self.lateral_axis)) 
    
                    context = "Right"
                    right_fs_times=list()
                    for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                        right_fs_times.append(ev.time())
    
                    logging.info("--right kinetic cycle--")
                    for i in range(0, len(right_fs_times)-1):
                        if right_fs_times[i] in times: 
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