# -*- coding: utf-8 -*-
import numpy as np
import logging


import pyCGM2.Tools.trialTools as CGM2trialTools
import pyCGM2.Math.normalisation  as MathNormalisation

import ma.io
import ma.body

#----module methods ------


def spatioTemporelParameter_descriptiveStats(cycles,label,context):

    """
        Compute descriptive statistics of spatio-temporal parameters from a `cycles` instance

        :Parameters:
             - `cycles` (pyCGM2.Processing.cycle.Cycles) - Cycles instance built fron CycleFilter
             - `label` (str) - spatio-temporal label
             - `context` (str) - cycle side context ( Left, Right)

        :Return:
            - `outDict` (dict)  - dictionnary with descriptive statistics ( mean, std, median).  Addictional Item *val* collects cycle values

    """

    outDict=dict()

    n=len([cycle for cycle in cycles if cycle.enableFlag and cycle.context==context])
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
        Compute descriptive statistics of point parameters from a `cycles` instance

        :Parameters:
             - `cycles` (pyCGM2.Processing.cycle.Cycles) - Cycles instance built fron CycleFilter
             - `label` (str) - point label
             - `context` (str) - cycle side context ( Left, Right)

        :Return:
            - `outDict` (dict)  - dictionnary with descriptive statistics ( mean, std, median).  Addictional Item *values* collects cycle values

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


    outDict = {'mean':meanData.reshape((101,1)), 'median':medianData.reshape((101,1)), 'std':stdData.reshape((101,1)), 'values': listOfPointValues }



    return outDict



def analog_descriptiveStats(cycles,label,context):
    """
        Compute descriptive statistics of analog parameters from a `cycles` instance

        :Parameters:
             - `cycles` (pyCGM2.Processing.cycle.Cycles) - Cycles instance built fron CycleFilter
             - `label` (str) - analog label
             - `context` (str) - cycle side context ( Left, Right)

        :Return:
            - `outDict` (dict)  - dictionnary with descriptive statistics ( mean, std, median).  Addictional Item *values* collects cycle values

    """



    outDict=dict()



    n=len([cycle for cycle in cycles if cycle.enableFlag and cycle.context==context]) # list comprehension , get number of enabled cycle

    listOfPointValues=list()
    x=np.empty((101,n))

    i=0
    for cycle in cycles:
        if cycle.enableFlag and cycle.context==context:
            tmp = cycle.getAnalogTimeSequenceDataNormalized(label)
            x[:,i]=tmp[:,0]
            i+=1
            listOfPointValues.append(tmp)

    #x_resize=x[0:1001:10,:]

    meanData=np.array(np.zeros((101,1)))
    meanData[:,0]=np.mean(x,axis=1)

    stdData=np.array(np.zeros((101,1)))
    stdData[:,0]=np.std(x,axis=1)

    medianData=np.array(np.zeros((101,1)))
    medianData[:,0]=np.median(x,axis=1)

    maximalValues = np.max(x,axis=0)

    outDict = {'mean':meanData, 'median':medianData, 'std':stdData, 'values': listOfPointValues, 'maxs': maximalValues }

    return outDict


def construcGaitCycle(trial):
    gaitCycles=list()

    context = "Left"
    left_fs_times=list()
    for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
        left_fs_times.append(ev.time())

    for i in range(0, len(left_fs_times)-1):
        gaitCycles.append (GaitCycle(trial, left_fs_times[i],left_fs_times[i+1],
                                       context))

    context = "Right"
    right_fs_times=list()
    for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
        right_fs_times.append(ev.time())

    for i in range(0, len(right_fs_times)-1):
        gaitCycles.append (GaitCycle(trial, right_fs_times[i],right_fs_times[i+1],
                                       context))

    return gaitCycles

#----module classes ------

class Cycle(ma.Node):
    """
        Cut out a trial and create a generic Cycle from specific times
    """
    #pour definir un label, il faut passer par la methode setName de l objet node



    def __init__(self,trial,startTime,endTime,context, enableFlag = True):
        """
        :Parameters:
             - `trial` (openma-trial) - openma from a c3d
             - `startTime` (double) -  start time of the cycle
             - `endTime` (double) - end time of the cycle
             - `enableFlag` (bool) - flag the Cycle in order to indicate if we can use it in a analysis process.

        .. note:

            no need Time sequence of type Analog
        """

        nodeLabel = "Cycle"
        super(Cycle,self).__init__(nodeLabel)
        self.trial=trial

        try:
            self.pointfrequency = trial.findChild(ma.T_TimeSequence,"",[["type",ma.TimeSequence.Type_Marker]]).sampleRate()
        except ValueError:
            raise Exception("[pyCGM2] : there are no time sequence of type marker in the openmaTrial")

        try:
            self.analogfrequency = trial.findChild(ma.T_TimeSequence,"",[["type",ma.TimeSequence.Type_Analog]]).sampleRate()
        except ValueError:
            self.analogfrequency = 0

        self.appf =  self.analogfrequency / self.pointfrequency
        self.firstFrame = int(round(trial.findChild(ma.T_TimeSequence,"",[["type",ma.TimeSequence.Type_Marker]]).startTime() * self.pointfrequency))


        self.begin =  int(round(startTime * self.pointfrequency) + 1)
        self.end = int(round(endTime * self.pointfrequency) + 1)
        self.context=context
        self.enableFlag = enableFlag

        self.discreteDataList=[]

        logging.debug("cycle makes from Frame %d   to  %d   (%s) " % (self.begin, self.end, self.context))



    def setEnableFlag(self,flag):
        """
        set the cycle flag

        :Parameters:
             - `flag` (bool) - boolean flag

        """
        self.enableFlag = flag

    def addDiscreteData(self,label,value,instant):
        pass #todo

    def getPointTimeSequenceData(self,pointLabel):
        """
            Get temporal point data of the cycle

            :Parameters:
                - `pointLabel` (str) - point Label

        """

        if CGM2trialTools.isTimeSequenceExist(self.trial,pointLabel):
            return self.trial.findChild(ma.T_TimeSequence, pointLabel).data()[self.begin-self.firstFrame:self.end-self.firstFrame+1,0:3] # 0.3 because openma::Ts includes a forth column (i.e residual)
        else:
            raise Exception("[pyCGM2] marker %s doesn t exist"% pointLabel )


    def getPointTimeSequenceDataNormalized(self,pointLabel):
        """
            Normalisation of a point label

            :Parameters:
                - `pointLabel` (str) - point Label

        """

        data = self.getPointTimeSequenceData(pointLabel)
        return  MathNormalisation.timeSequenceNormalisation(101,data)

    def getAnalogTimeSequenceData(self,analogLabel):
        """
            Get analog data of the cycle

            :Parameters:
                - `analogLabel` (str) - analog Label

        """

        if CGM2trialTools.isTimeSequenceExist(self.trial,analogLabel):
            return  self.trial.findChild(ma.T_TimeSequence, analogLabel).data()[int((self.begin-self.firstFrame) * self.appf) : int((self.end-self.firstFrame+1) * self.appf),:]
        else:
            raise Exception("[pyCGM2] Analog %s doesn t exist"% analogLabel )

    def getAnalogTimeSequenceDataNormalized(self,analogLabel):
        """
            Get analog data of the cycle

            :Parameters:
                - `analogLabel` (str) - analog Label
        """

        data = self.getAnalogTimeSequenceData(analogLabel)
        return  MathNormalisation.timeSequenceNormalisation(101,data)

    def getEvents(self,context="All"):
        """
            Get all events of the cycle

            :Parameters:
                - `context` (str) - event context ( All, Left or Right)

        """

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

    """
        Herited method of Cycle specifying a Gait Cycle

        .. note::

            - By default, X0 and Yo are the longitudinal and lateral global axis respectively
            - Each GaitCycle creation computes spatio-temporal parameters automatically.
            - spatio-temporal parameters are :
                "duration", "cadence",
                "stanceDuration", "stancePhase",
                "swingDuration", "swingPhase", "doubleStance1", "doubleStance2",
                "simpleStance", "strideLength", "stepLength",
                "strideWidth", "speed"
    """


    STP_LABELS=["duration","cadence", "stanceDuration",  "stancePhase",
                "swingDuration", "swingPhase", "doubleStance1", "doubleStance2",
                "simpleStance", "strideLength", "stepLength",
                "strideWidth", "speed"]


    def __init__(self,gaitTrial,startTime,endTime,context, enableFlag = True):
        """
        :Parameters:
             - `trial` (openma-trial) - openma from a c3d
             - `startTime` (double) -  start time of the cycle
             - `endTime` (double) - end time of the cycle
             - `enableFlag` (bool) - flag the Cycle in order to indicate if we can use it in a analysis process.

        """



        super(GaitCycle, self).__init__(gaitTrial,startTime,endTime,context, enableFlag = enableFlag)

        #ajout des oppositeFO, contraFO,oopositeFS
        evs=self.getEvents()
        if context=="Right":
            oppositeSide="Left"
        elif context=="Left":
            oppositeSide="Right"
        for ev in evs.findChildren(ma.T_Event):
            if ev.name() == "Foot Off" and ev.context()==oppositeSide:
                oppositeFO= int(round(ev.time() * self.pointfrequency) + 1)
            if ev.name() == "Foot Strike" and ev.context()==oppositeSide:
                oppositeFS= int(round(ev.time() * self.pointfrequency) + 1)
            if ev.name() == "Foot Off" and ev.context()==context:
                contraFO = int(round(ev.time() * self.pointfrequency) + 1)
        if oppositeFO > oppositeFS:
            raise Exception("[pyCGM2] : check your c3d - Gait event error")


        self.m_oppositeFO=oppositeFO
        self.m_oppositeFS=oppositeFS
        self.m_contraFO=contraFO
        self.m_normalizedOppositeFO=round(np.divide(float(self.m_oppositeFO - self.begin),float(self.end-self.begin))*100)
        self.m_normalizedOppositeFS=round(np.divide(float(self.m_oppositeFS - self.begin),float(self.end-self.begin))*100)
        self.m_normalizedContraFO=round(np.divide(float(self.m_contraFO - self.begin),float(self.end-self.begin))*100)


        self.__computeSpatioTemporalParameter()


    def __computeSpatioTemporalParameter(self):

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


                progressionAxis,forwardProgression,globalFrame = CGM2trialTools.findProgression(self.trial,"LHEE")

                longitudinal_axis=0  if progressionAxis =="X" else 1
                lateral_axis=1  if progressionAxis =="X" else 0


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

                progressionAxis,forwardProgression,globalFrame = CGM2trialTools.findProgression(self.trial,"RHEE")

                longitudinal_axis=0  if progressionAxis =="X" else 1
                lateral_axis=1  if progressionAxis =="X" else 0

                strideLength=np.abs(self.getPointTimeSequenceData("RHEE")[self.end-self.begin,longitudinal_axis] -\
                                    self.getPointTimeSequenceData("RHEE")[0,longitudinal_axis])/1000.0

                strideWidth = np.abs(self.getPointTimeSequenceData("RTOE")[self.end-self.begin,lateral_axis] -\
                                 self.getPointTimeSequenceData("LHEE")[0,lateral_axis])/1000.0

                pst.setProperty("strideLength", strideLength)
                pst.setProperty("strideWidth", strideWidth)

                stepLength = np.abs(self.getPointTimeSequenceData("RHEE")[self.m_oppositeFS-self.begin,longitudinal_axis] -\
                                    self.getPointTimeSequenceData("LHEE")[0,longitudinal_axis])/1000.0
                pst.setProperty("stepLength", stepLength)

                pst.setProperty("speed",np.divide(strideLength,duration))

    def getSpatioTemporalParameter(self,label):
        """
        Return a spatio-temporal parameter

        :Parameters:
             - `label` (str) - label of the desired spatio-temporal parameter
        """

        stpNode = self.findChild(ma.T_Node,"stp")
        return stpNode.property(label).cast()


# ----- PATTERN BUILDER -----

# ----- FILTER -----
class CyclesFilter:
    """
        Filter buiding a Cycles instance.
    """

    __builder = None

    def setBuilder(self, builder):
        """
            Set the cycle builder

            :Parameters:
                 - `builder` (concrete cycleBuilder) - a concrete cycle builder

        """
        self.__builder = builder


    def build(self):
        """
            Build a `Cycles` instance
        """
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


# --- OBJECT TO BUILD
class Cycles():
    """
    Object to build from CycleFilter.

    Cycles work as **class-container**. Its attribute member collects list of build Cycle

    Attribute :

      - `spatioTemporalCycles` - (list of Cycle) - collect list of build cycles uses for spatiotemporal analysis
      - `kinematicCycles` - (list of Cycle) - collect list of build cycles uses for kinematic analysis
      - `kineticCycles` - (list of Cycle) - collect list of build cycles uses for kinetic analysis
      - `emgCycles` - (list of Cycle) - collect list of build cycles uses for emg analysis

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


# --- BUILDER
class CyclesBuilder(object):

    def __init__(self,spatioTemporalTrials=None,kinematicTrials=None,kineticTrials=None,emgTrials=None):

        self.spatioTemporalTrials =spatioTemporalTrials
        self.kinematicTrials =kinematicTrials
        self.kineticTrials =kineticTrials
        self.emgTrials =emgTrials

    def getSpatioTemporal(self):

        if self.spatioTemporalTrials is not None:
            spatioTemporalCycles=list()
            for trial in  self.spatioTemporalTrials:

                context = "Left"
                left_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                    left_fs_times.append(ev.time())

                for i in range(0, len(left_fs_times)-1):
                    spatioTemporalCycles.append (Cycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                   context))

                context = "Right"
                right_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                    right_fs_times.append(ev.time())

                for i in range(0, len(right_fs_times)-1):
                    spatioTemporalCycles.append (Cycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                   context))

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
                    kinematicCycles.append (Cycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                   context))

                context = "Right"
                right_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                    right_fs_times.append(ev.time())

                for i in range(0, len(right_fs_times)-1):
                    kinematicCycles.append (Cycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                   context))

            return kinematicCycles
        else:
            return None

    def getKinetics(self):

        if self.kineticTrials is not None:

            detectionTimeOffset = 0.02

            kineticCycles=list()
            for trial in  self.kineticTrials:

                flag_kinetics,times,times_left,times_right = CGM2trialTools.isKineticFlag(trial)

                if flag_kinetics:
                    context = "Left"
                    count_L=0
                    left_fs_times=list()
                    for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                        left_fs_times.append(ev.time())

                    for i in range(0, len(left_fs_times)-1):
                        init =  left_fs_times[i]
                        end =  left_fs_times[i+1]

                        for timeKinetic in times_left:

                            if timeKinetic<=end and timeKinetic>=init:
                                logging.debug("Left kinetic cycle found from %.2f to %.2f" %(left_fs_times[i], left_fs_times[i+1]))
                                kineticCycles.append (Cycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                           context))

                                count_L+=1
                    logging.debug("%i Left Kinetic cycles available" %(count_L))



                    context = "Right"
                    count_R=0
                    right_fs_times=list()
                    for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                        right_fs_times.append(ev.time())

                    for i in range(0, len(right_fs_times)-1):
                        init =  right_fs_times[i]
                        end =  right_fs_times[i+1]

                        for timeKinetic in times_right:
                            if timeKinetic<=end and timeKinetic>=init:
                                logging.debug("Right kinetic cycle found from %.2f to %.2f" %(right_fs_times[i], right_fs_times[i+1]))
                                kineticCycles.append (Cycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                           context))
                                count_R+=1
                    logging.debug("%i Right Kinetic cycles available" %(count_R))

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
                    emgCycles.append (Cycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                   context))

                context = "Right"
                right_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                    right_fs_times.append(ev.time())

                for i in range(0, len(right_fs_times)-1):
                    emgCycles.append (Cycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                   context))

            return emgCycles
        else:
            return None

class GaitCyclesBuilder(CyclesBuilder):
    """
        Concrete builder extracting Cycles from gait trials

        .. important:: The underlying concept is spatio-temporal parameters, kinematic outputs, kinetic ouputs or emg  could be extracted from different gait trials.

    """

    def __init__(self,spatioTemporalTrials=None,kinematicTrials=None,kineticTrials=None,emgTrials=None):
        """
            :Parameters:
                 - `spatioTemporalTrials` (list of openma trials) - list of trials of which Cycles will be extracted for computing spatio-temporal parameters
                 - `kinematicTrials` (list of openma trials) - list of trials of which Cycles will be extracted for computing kinematic outputs
                 - `kineticTrials` (list of openma trials) - list of trials of which Cycles will be extracted for computing kinetic outputs
                 - `emgTrials` (list of openma trials) - list of trials of which Cycles will be extracted for emg
                 - `longitudinal_axis` (str) - label of the  longitudinal global axis (X, Y or Z)
                 - `lateral_axis` (str) - label of the  longitudinal global axis (X, Y or Z)

        """


        super(GaitCyclesBuilder, self).__init__(
            spatioTemporalTrials = spatioTemporalTrials,
            kinematicTrials=kinematicTrials,
            kineticTrials = kineticTrials,
            emgTrials = emgTrials,
            )


    def getSpatioTemporal(self):
        """
           extract Cycles used for  spatio Temporal parameters

           :return:
               -`spatioTemporalCycles` (list of GaitCycle)
        """

        if self.spatioTemporalTrials is not None:
            spatioTemporalCycles=list()
            for trial in  self.spatioTemporalTrials:

                context = "Left"
                left_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                    left_fs_times.append(ev.time())

                for i in range(0, len(left_fs_times)-1):
                    spatioTemporalCycles.append (GaitCycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                   context))

                context = "Right"
                right_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                    right_fs_times.append(ev.time())

                for i in range(0, len(right_fs_times)-1):
                    spatioTemporalCycles.append (GaitCycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                   context))

            return spatioTemporalCycles
        else:
            return None

    def getKinematics(self):
        """
           extract Cycles used for kinematic outputs

           :return:
             -`kinematicCycles` (list of GaitCycle)
        """

        if self.kinematicTrials is not None:
            kinematicCycles=list()
            for trial in  self.kinematicTrials:

                context = "Left"
                left_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                    left_fs_times.append(ev.time())

                for i in range(0, len(left_fs_times)-1):
                    kinematicCycles.append (GaitCycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                   context))

                context = "Right"
                right_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                    right_fs_times.append(ev.time())

                for i in range(0, len(right_fs_times)-1):
                    kinematicCycles.append (GaitCycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                   context))

            return kinematicCycles
        else:
            return None

    def getKinetics(self):
        if self.kineticTrials is not None:

            detectionTimeOffset = 0.02

            kineticCycles=list()
            for trial in  self.kineticTrials:

                flag_kinetics,times,times_left,times_right = CGM2trialTools.isKineticFlag(trial)

                if flag_kinetics:
                    context = "Left"
                    count_L=0
                    left_fs_times=list()
                    for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                        left_fs_times.append(ev.time())

                    for i in range(0, len(left_fs_times)-1):
                        init =  left_fs_times[i]
                        end =  left_fs_times[i+1]

                        for timeKinetic in times_left:

                            if timeKinetic<=end and timeKinetic>=init:
                                logging.debug("Left kinetic cycle found from %.2f to %.2f" %(left_fs_times[i], left_fs_times[i+1]))
                                kineticCycles.append (GaitCycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                           context))

                                count_L+=1
                    logging.debug("%i Left Kinetic cycles available" %(count_L))



                    context = "Right"
                    count_R=0
                    right_fs_times=list()
                    for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                        right_fs_times.append(ev.time())

                    for i in range(0, len(right_fs_times)-1):
                        init =  right_fs_times[i]
                        end =  right_fs_times[i+1]

                        for timeKinetic in times_right:
                            if timeKinetic<=end and timeKinetic>=init:
                                logging.debug("Right kinetic cycle found from %.2f to %.2f" %(right_fs_times[i], right_fs_times[i+1]))
                                kineticCycles.append (GaitCycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                           context))
                                count_R+=1
                    logging.debug("%i Right Kinetic cycles available" %(count_R))

            return kineticCycles
        else:
            return None

    def getEmg(self):
        """
            Extract Cycles used for emg

            :return:
                -`emgCycles` (list of GaitCycle)
        """

        if self.emgTrials is not None:
            emgCycles=list()
            for trial in  self.emgTrials:

                context = "Left"
                left_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                    left_fs_times.append(ev.time())

                for i in range(0, len(left_fs_times)-1):
                    emgCycles.append (GaitCycle(trial, left_fs_times[i],left_fs_times[i+1],
                                                   context))

                context = "Right"
                right_fs_times=list()
                for ev in  trial.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Right"]]):
                    right_fs_times.append(ev.time())

                for i in range(0, len(right_fs_times)-1):
                    emgCycles.append (GaitCycle(trial, right_fs_times[i],right_fs_times[i+1],
                                                   context))

            return emgCycles
        else:
            return None
