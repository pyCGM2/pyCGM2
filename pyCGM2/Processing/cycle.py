# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

"""

This module aims to construct a `Cycles` instance. Based on a *builder pattern design*,
the Filter `CyclesFilter` calls a builder, ie `CyclesBuilder` or
a `GaitCyclesBuilder` in the context of gait Analysis, then return a `Cycles` instance.

As attributes,  the  `Cycles` instance distinguished series of `Cycle` (or `GaitCycle`) instance
according to computational objectives ( ie computation of spatio-temporal parameters, kinematics,
kinetics or emg.)

"""
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER

from pyCGM2.Processing.ProgressionFrame import progressionFrameFilters
from pyCGM2.Processing.ProgressionFrame import progressionFrameProcedures

import pyCGM2.Math.normalisation as MathNormalisation

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")

from pyCGM2.Utils import utils
from pyCGM2.Tools import btkTools


def spatioTemporelParameter_descriptiveStats(cycles,label,context):

    """Compute descriptive statistics of spatio-temporal parameters from a `cycles` instance

    Args:
        cycles (pyCGM2.Processing.cycle.Cycles): Cycles instance
        label (str): spatio-temporal label
        context (str): event context ( Left, Right)
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
    """Compute descriptive statistics of point parameters from a `cycles` instance

    Args:
        cycles (pyCGM2.Processing.cycle.Cycles): Cycles instance
        label (str): point label
        context (str): event context ( Left, Right)

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
    stdData=np.array(np.zeros((101,3)))
    medianData=np.array(np.zeros((101,3)))

    if not np.all(x==0):
        x[x == 0] = np.nan
        meanData[:,0] = np.nanmean(x, axis=1)
        stdData[:,0]=np.nanstd(x,axis=1)
        medianData[:,0]=np.nanmedian(x,axis=1)

    if not np.all(y==0):
        y[y == 0] = np.nan
        meanData[:,1] = np.nanmean(y, axis=1)
        stdData[:,1]=np.nanstd(y,axis=1)
        medianData[:,1]=np.nanmedian(y,axis=1)


    if not np.all(z==0):
        z[z == 0] = np.nan
        meanData[:,2] = np.nanmean(z, axis=1)
        stdData[:,2]=np.nanstd(z,axis=1)
        medianData[:,2]=np.nanmedian(z,axis=1)


    outDict = {'mean':meanData, 'median':medianData, 'std':stdData, 'values': listOfPointValues }



    return outDict



def analog_descriptiveStats(cycles,label,context):
    """Compute descriptive statistics of analog parameters from a `cycles` instance

    Args:
        cycles (pyCGM2.Processing.cycle.Cycles): Cycles instance
        label (str): analog label
        context (str): event context ( Left, Right)
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
            listOfPointValues.append(tmp)
            i+=1


    meanData=np.array(np.zeros((101,1)))
    if not np.all(x==0):
        meanData[:,0]=np.nanmean(x,axis=1)

    stdData=np.array(np.zeros((101,1)))
    if not np.all(x==0):
        stdData[:,0]=np.nanstd(x,axis=1)

    medianData=np.array(np.zeros((101,1)))
    if not np.all(x==0):
        medianData[:,0]=np.nanmedian(x,axis=1)

    maximalValues = np.max(x,axis=0)

    outDict = {'mean':meanData, 'median':medianData, 'std':stdData, 'values': listOfPointValues, 'maxs': maximalValues}

    return outDict


def construcGaitCycle(acq):
    """Construct gait cycle

    Args:
        acq (btk.Acquisition): an acquisition

    """

    gaitCycles=list()

    context = "Left"
    left_fs_frames=list()
    for ev in btk.Iterate(acq.GetEvents()):
        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
            left_fs_frames.append(ev.GetFrame())

    for i in range(0, len(left_fs_frames)-1):
        gaitCycles.append (GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                       context))

    context = "Right"
    right_fs_frames=list()
    for ev in btk.Iterate(acq.GetEvents()):
        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
            right_fs_frames.append(ev.GetFrame())

    for i in range(0, len(right_fs_frames)-1):
        gaitCycles.append (GaitCycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                       context))

    return gaitCycles



class Cycle(object):
    """ Generic Cycle

    Args:
        acq (btk.Acquisituion): an acquisition
        startFrame (int):  start frame of the cycle
        endFrame (double): end frame of the cycle
        enableFlag (bool): flag  to indicate the cycle will be use in further computation.

    """


    def __init__(self,acq,startFrame,endFrame,context, enableFlag = True):

        self.acq=acq

        self.pointfrequency = float(acq.GetPointFrequency())
        self.analogfrequency = float(acq.GetAnalogFrequency())

        self.appf =  self.analogfrequency / self.pointfrequency
        self.firstFrame = acq.GetFirstFrame()

        self.begin =  startFrame
        self.end = endFrame
        self.context=context
        self.enableFlag = enableFlag

        self.discreteDataList=[]

        self.stps =dict()

        LOGGER.logger.debug("cycle makes from Frame %d   to  %d   (%s) " % (self.begin, self.end, self.context))



    def setEnableFlag(self,flag):
        """enable/disable the cycle

        Args:
             flag (bool): boolean flag

        """
        self.enableFlag = flag

    def addDiscreteData(self,label,value,instant):
        pass

    def getPointTimeSequenceData(self,pointLabel):
        """Get temporal point data of the cycle

        Args:
            pointLabel (str): point Label

        """

        if btkTools.isPointExist(self.acq,pointLabel):
            return self.acq.GetPoint(pointLabel).GetValues()[self.begin-self.firstFrame:self.end-self.firstFrame+1,0:3] # 0.3 because openma::Ts includes a forth column (i.e residual)
        else:
            LOGGER.logger.debug("[pyCGM2] the point Label %s doesn t exist " % (pointLabel))
            return None



    def getPointTimeSequenceDataNormalized(self,pointLabel):
        """Time-Normalise  a point label

        Args:
            pointLabel (str): point Label

        """

        data = self.getPointTimeSequenceData(pointLabel)
        if data is None:
            out=np.zeros((101,3))
        else:
            out = MathNormalisation.timeSequenceNormalisation(101,data)

        return out

    def getAnalogTimeSequenceData(self,analogLabel):
        """Get analog data of the cycle

        Args:
            analogLabel (str): analog Label

        """
        if btkTools.isAnalogExist(self.acq,analogLabel):
            return  self.acq.GetAnalog(analogLabel).GetValues()[int((self.begin-self.firstFrame) * self.appf) : int((self.end-self.firstFrame+1) * self.appf),:]
        else:
            LOGGER.logger.debug("[pyCGM2] the Analog Label %s doesn t exist" % (analogLabel))
            return None


    def getAnalogTimeSequenceDataNormalized(self,analogLabel):
        """Time-normalize an analog data

        Args:
            analogLabel (str): analog Label
        """

        data = self.getAnalogTimeSequenceData(analogLabel)
        if data is None:
            out=np.zeros((101,3))
        else:
            out = MathNormalisation.timeSequenceNormalisation(101,data)

        return  out

    def getEvents(self,context="All"):
        """Get all events of the cycle

        Args:
            context (str): event context (All, Left or Right)

        """
        events = list()
        evsn = self.acq.GetEvents()
        for ev in btk.Iterate(evsn):
            if context==("Left" or "Right") :
                if ev.GetContext()== context:
                    if ev.GetFrame() + 1  >self.begin and  ev.GetFrame() + 1<self.end:
                        events.append(ev)
            else:
                if ev.GetFrame() + 1  >self.begin and  ev.GetFrame() + 1<self.end:
                    events.append(ev)
        return events


class GaitCycle(Cycle):

    """Inherited class from Cycle defining a Gait cycle.

    Args:
        acq (btk.Acquisituion): an acquisition
        startFrame (int):  start frame of the cycle
        endFrame (double): end frame of the cycle
        enableFlag (bool): flag  to indicate the cycle will be use in further computation.

    **Notes**

        - By default, X0 and Y0 are the longitudinal and lateral global axis respectively
        - `GaitCycle` construction computes spatio-temporal parameters automatically.
        - spatio-temporal parameters are :
            "duration", "cadence",
            "stanceDuration", "stepDuration", "doubleStance1Duration",
            "doubleStance2Duration", "simpleStanceDuration", "stancePhase",
            "swingDuration", "swingPhase", "doubleStance1", "doubleStance2",
            "simpleStance", "stepPhase", "strideLength", "stepLength",
            "strideWidth", "speed"
    """


    STP_LABELS=["duration","cadence",
                "stanceDuration", "stepDuration", "doubleStance1Duration",
                "doubleStance2Duration","simpleStanceDuration","stancePhase",
                "swingDuration", "swingPhase", "doubleStance1", "doubleStance2",
                "simpleStance", "stepPhase","strideLength", "stepLength",
                "strideWidth", "speed"]


    def __init__(self,gaitAcq,startFrame,endFrame,context, enableFlag = True):
        super(GaitCycle, self).__init__(gaitAcq,startFrame,endFrame,context, enableFlag = enableFlag)

        evs=self.getEvents()

        if context=="Right":
            oppositeSide="Left"
        elif context=="Left":
            oppositeSide="Right"
        for ev in evs:
            if ev.GetLabel() == "Foot Off" and ev.GetContext()==oppositeSide:
                oppositeFO= ev.GetFrame()
            if ev.GetLabel() == "Foot Strike" and ev.GetContext()==oppositeSide:
                oppositeFS= ev.GetFrame()
            if ev.GetLabel() == "Foot Off" and ev.GetContext()==context:
                contraFO = ev.GetFrame()
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
        stanceDuration=np.divide(np.abs(self.m_contraFO - self.begin) , self.pointfrequency)
        swingDuration=np.divide(np.abs(self.m_contraFO - self.end) , self.pointfrequency)
        stepDuration=np.divide(np.abs(self.m_oppositeFS - self.begin) , self.pointfrequency)

        self.stps["duration"] = duration
        self.stps["cadence"]= np.divide(60.0,duration)

        self.stps["stanceDuration"] = stanceDuration
        self.stps["swingDuration"] =  swingDuration
        self.stps["stepDuration"] =  stepDuration
        self.stps["doubleStance1Duration"] =  np.divide(np.abs(self.m_oppositeFO - self.begin) , self.pointfrequency)
        self.stps["doubleStance2Duration"] =  np.divide(np.abs(self.m_contraFO - self.m_oppositeFS) , self.pointfrequency)
        self.stps["simpleStanceDuration"] =  np.divide(np.abs(self.m_oppositeFO - self.m_oppositeFS) , self.pointfrequency)


        self.stps["stancePhase"] =  round(np.divide(stanceDuration,duration)*100)
        self.stps["swingPhase"] =  round(np.divide(swingDuration,duration)*100 )
        self.stps["doubleStance1"] =  round(np.divide(np.divide(np.abs(self.m_oppositeFO - self.begin) , self.pointfrequency),duration)*100)
        self.stps["doubleStance2"] =  round(np.divide(np.divide(np.abs(self.m_contraFO - self.m_oppositeFS) , self.pointfrequency),duration)*100)
        self.stps["simpleStance"] =  round(np.divide(np.divide(np.abs(self.m_oppositeFO - self.m_oppositeFS) , self.pointfrequency),duration)*100)
        self.stps["stepPhase"] =  round(np.divide(stepDuration,duration)*100)

        if self.context == "Left":

            if btkTools.isPointExist(self.acq,"LHEE") and btkTools.isPointExist(self.acq,"RHEE") and btkTools.isPointExist(self.acq,"LTOE"):

                pfp = progressionFrameProcedures.PointProgressionFrameProcedure(marker="LHEE")
                pff = progressionFrameFilters.ProgressionFrameFilter(self.acq,pfp)
                pff.compute()
                progressionAxis =  pff.outputs["progressionAxis"]
                forwardProgression = pff.outputs["forwardProgression"]
                globalFrame = pff.outputs["globalFrame"]


                longitudinal_axis=0  if progressionAxis =="X" else 1
                lateral_axis=1  if progressionAxis =="X" else 0


                strideLength=np.abs(self.getPointTimeSequenceData("LHEE")[self.end-self.begin,longitudinal_axis] -\
                                    self.getPointTimeSequenceData("LHEE")[0,longitudinal_axis])/1000.0
                self.stps["strideLength"] =  strideLength

                stepLength = np.abs(self.getPointTimeSequenceData("RHEE")[self.m_oppositeFS-self.begin,longitudinal_axis] -\
                                    self.getPointTimeSequenceData("LHEE")[0,longitudinal_axis])/1000.0
                self.stps["stepLength"] =  stepLength

                strideWidth = np.abs(self.getPointTimeSequenceData("LTOE")[self.end-self.begin,lateral_axis] -\
                                     self.getPointTimeSequenceData("RHEE")[0,lateral_axis])/1000.0
                self.stps["strideWidth"] =  strideWidth

                self.stps["speed"] = np.divide(strideLength,duration)


        if self.context == "Right":

            if btkTools.isPointExist(self.acq,"RHEE") and btkTools.isPointExist(self.acq,"LHEE") and btkTools.isPointExist(self.acq,"RTOE"):

                pfp = progressionFrameProcedures.PointProgressionFrameProcedure(marker="RHEE")
                pff = progressionFrameFilters.ProgressionFrameFilter(self.acq,pfp)
                pff.compute()
                progressionAxis =  pff.outputs["progressionAxis"]
                forwardProgression = pff.outputs["forwardProgression"]
                globalFrame = pff.outputs["globalFrame"]

                longitudinal_axis=0  if progressionAxis =="X" else 1
                lateral_axis=1  if progressionAxis =="X" else 0

                strideLength=np.abs(self.getPointTimeSequenceData("RHEE")[self.end-self.begin,longitudinal_axis] -\
                                    self.getPointTimeSequenceData("RHEE")[0,longitudinal_axis])/1000.0

                strideWidth = np.abs(self.getPointTimeSequenceData("RTOE")[self.end-self.begin,lateral_axis] -\
                                 self.getPointTimeSequenceData("LHEE")[0,lateral_axis])/1000.0

                self.stps["strideLength"] =  strideLength
                self.stps["strideWidth"] =  strideWidth

                stepLength = np.abs(self.getPointTimeSequenceData("RHEE")[self.m_oppositeFS-self.begin,longitudinal_axis] -\
                                    self.getPointTimeSequenceData("LHEE")[0,longitudinal_axis])/1000.0
                self.stps["stepLength"] =  stepLength

                self.stps["speed"] = np.divide(strideLength,duration)

    def getSpatioTemporalParameter(self,label):
        """ Return a spatio-temporal parameter

        Args:
             label (str): label of the desired spatio-temporal parameter
        """

        return self.stps[label]


# ----- PATTERN BUILDER -----

# ----- FILTER -----
class CyclesFilter:
    """ Filter buiding a `Cycles` instance.
    """

    __builder = None

    def setBuilder(self, builder):
        """Set the builder

        Args:
            builder (pyCGM2.Processing.Cycle.(Builder)): a concrete cycle builder

        """
        self.__builder = builder


    def build(self):
        """ Build a `Cycles` instance
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



class Cycles():
    """
    Object to build from CycleFilter.

    Cycles work as **class-container**. Its attribute members collect list of `Cycle`
    or `GaitCycle` according to computational objectives

    **Attributes** are

      - spatioTemporalCycles:  list of cycles uses for spatiotemporal parameter computation
      - kinematicCycles: list of cycles uses for kinematic computation
      - kineticCycles: list of cycles uses for kinetic computation
      - emgCycles: list of cycles uses for emg computation

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
    """Builder of generic cycles

    Args:
        spatioTemporalAcqs (list): acquisitions used for  spatio-temporal parameter computation
        kinematicAcqs (list):acquisitions used for  kinematics computation
        kineticAcqs (list): acquisitions used for kinetics computation
        emgAcqs (list): acquisitions used for emg computation

    """

    def __init__(self,spatioTemporalAcqs=None,kinematicAcqs=None,kineticAcqs=None,emgAcqs=None):

        self.spatioTemporalAcqs =spatioTemporalAcqs
        self.kinematicAcqs =kinematicAcqs
        self.kineticAcqs =kineticAcqs
        self.emgAcqs =emgAcqs

    def getSpatioTemporal(self):
        """get the list of Cycle used for  spatio-temporal parameter computation
        """

        if self.spatioTemporalAcqs is not None:
            spatioTemporalCycles=list()
            for acq in  self.spatioTemporalAcqs:

                startFrame = acq.GetFirstFrame()
                endFrame = acq.GetLastFrame()

                context = "Left"
                left_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())
                #
                # for ev in  acq.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                #     left_fs_frames.append(ev.time())

                context = "Right"
                right_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        right_fs_frames.append(ev.GetFrame())


                if left_fs_frames == [] and  right_fs_frames == []:
                    spatioTemporalCycles.append (Cycle(acq, startFrame,endFrame,"Left"))
                    spatioTemporalCycles.append (Cycle(acq, startFrame,endFrame,"Right"))
                    LOGGER.logger.info("[pyCGM2] left and Right context - time normalization from time boudaries")

                if len(left_fs_frames) >1:
                    for i in range(0, len(left_fs_frames)-1):
                        spatioTemporalCycles.append (Cycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                       "Left"))
                elif len(left_fs_frames) ==1:
                    LOGGER.logger.warning("[pyCGM2] No left cycles, only one left foot strike detected)")
                else:
                    LOGGER.logger.warning("[pyCGM2] No left cycles")

                if len(right_fs_frames)>1:
                    for i in range(0, len(right_fs_frames)-1):
                        spatioTemporalCycles.append (Cycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                       "Right"))
                elif len(right_fs_frames) ==1:
                    LOGGER.logger.warning("[pyCGM2] No right cycles, only one right foot strike detected)")
                else:
                    LOGGER.logger.warning("[pyCGM2] No Right cycles")

            return spatioTemporalCycles
        else:
            return None

    def getKinematics(self):
        """get the list of Cycle used for  kinematic computation
        """
        if self.kinematicAcqs is not None:
            kinematicCycles=list()
            for acq in  self.kinematicAcqs:

                startFrame = acq.GetFirstFrame()
                endFrame = acq.GetLastFrame()

                context = "Left"
                left_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())


                context = "Right"
                right_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        right_fs_frames.append(ev.GetFrame())

                if left_fs_frames == [] and  right_fs_frames == []:
                    kinematicCycles.append (Cycle(acq, startFrame,endFrame,"Left"))
                    kinematicCycles.append (Cycle(acq, startFrame,endFrame,"Right"))
                    LOGGER.logger.info("[pyCGM2] left and Right context - time normalization from time boudaries")

                if len(left_fs_frames) >1:
                    for i in range(0, len(left_fs_frames)-1):
                        kinematicCycles.append (Cycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                       "Left"))
                elif len(left_fs_frames) ==1:
                    LOGGER.logger.warning("[pyCGM2] No left cycles, only one left foot strike detected)")
                else:
                    LOGGER.logger.warning("[pyCGM2] No left cycles")

                if len(right_fs_frames) >1:
                    for i in range(0, len(right_fs_frames)-1):
                        kinematicCycles.append (Cycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                       "Right"))
                elif len(right_fs_frames) ==1:
                    LOGGER.logger.warning("[pyCGM2] No right cycles, only one right foot strike detected)")
                else:
                    LOGGER.logger.warning("[pyCGM2] No Right cycles")

            return kinematicCycles
        else:
            return None

    def getKinetics(self):
        """get the list of Cycle used for  kinetic computation
        """
        if self.kineticAcqs is not None:

            detectionTimeOffset = 0.02

            kineticCycles=list()
            for acq in  self.kineticAcqs:

                flag_kinetics,frames,frames_left,frames_right = btkTools.isKineticFlag(acq)

                startFrame = acq.GetFirstFrame()
                endFrame = acq.GetLastFrame()

                if flag_kinetics:

                    context = "Left"
                    left_fs_frames=list()
                    for ev in btk.Iterate(acq.GetEvents()):
                        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                            left_fs_frames.append(ev.GetFrame())

                    context = "Right"
                    right_fs_frames=list()
                    for ev in btk.Iterate(acq.GetEvents()):
                        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                            right_fs_frames.append(ev.GetFrame())


                    if left_fs_frames == [] and right_fs_frames==[]:
                        kineticCycles.append (Cycle(acq, startFrame,endFrame,"Left"))
                        kineticCycles.append (Cycle(acq, startFrame,endFrame,"Right"))
                        LOGGER.logger.info("[pyCGM2] left - time normalization from time boudaries")


                    count_L=0
                    if len(left_fs_frames)>1:
                        for i in range(0, len(left_fs_frames)-1):
                            init =  left_fs_frames[i]
                            end =  left_fs_frames[i+1]

                            for frameKinetic in frames_left:

                                if frameKinetic<=end and frameKinetic>=init:
                                    LOGGER.logger.debug("Left kinetic cycle found from %.2f to %.2f" %(left_fs_frames[i], left_fs_frames[i+1]))
                                    kineticCycles.append (Cycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                               "Left"))

                                    count_L+=1
                        LOGGER.logger.debug("%i Left Kinetic cycles available" %(count_L))
                    elif len(left_fs_frames) ==1:
                        LOGGER.logger.warning("[pyCGM2] No left cycles, only one left foot strike detected)")
                    else:
                        LOGGER.logger.warning("[pyCGM2] No left cycles")

                    count_R=0
                    if len(right_fs_frames)>1:
                        for i in range(0, len(right_fs_frames)-1):
                            init =  right_fs_frames[i]
                            end =  right_fs_frames[i+1]

                            for frameKinetic in frames_right:
                                if frameKinetic<=end and frameKinetic>=init:
                                    LOGGER.logger.debug("Right kinetic cycle found from %.2f to %.2f" %(right_fs_frames[i], right_fs_frames[i+1]))
                                    kineticCycles.append (Cycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                               "Right"))
                                    count_R+=1
                        LOGGER.logger.debug("%i Right Kinetic cycles available" %(count_R))
                    elif len(right_fs_frames) ==1:
                        LOGGER.logger.warning("[pyCGM2] No right cycles, only one right foot strike detected)")
                    else:
                        LOGGER.logger.warning("[pyCGM2] No Right cycles")

            return kineticCycles
        else:
            return None

    def getEmg(self):
        """get the list of Cycle used for emg computation
        """
        if self.emgAcqs is not None:
            emgCycles=list()
            for acq in  self.emgAcqs:

                startFrame = acq.GetFirstFrame()
                endFrame = acq.GetLastFrame()

                context = "Left"
                left_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())

                context = "Right"
                right_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        right_fs_frames.append(ev.GetFrame())


                if left_fs_frames == [] and  right_fs_frames == []:
                    emgCycles.append (Cycle(acq, startFrame,endFrame,"Left"))
                    emgCycles.append (Cycle(acq, startFrame,endFrame,"Right"))
                    LOGGER.logger.info("[pyCGM2] left and Right context - time normalization from time boudaries")

                if len(left_fs_frames)>1:
                    for i in range(0, len(left_fs_frames)-1):
                        emgCycles.append (Cycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                       "Left"))
                elif len(left_fs_frames) ==1:
                    LOGGER.logger.warning("[pyCGM2] No left cycles, only one left foot strike detected)")
                else:
                    LOGGER.logger.warning("[pyCGM2] No left cycles")

                if len(right_fs_frames)>1:
                    for i in range(0, len(right_fs_frames)-1):
                        emgCycles.append (Cycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                       "Right"))
                elif len(right_fs_frames) ==1:
                    LOGGER.logger.warning("[pyCGM2] No right cycles, only one right foot strike detected)")
                else:
                    LOGGER.logger.warning("[pyCGM2] No Right cycles")

            return emgCycles

        else:
            return None

class GaitCyclesBuilder(CyclesBuilder):
    """ Builder of gait cycle

    Args:
        spatioTemporalAcqs (list): acquisitions used for  spatio-temporal parameter computation
        kinematicAcqs (list):acquisitions used for  kinematics computation
        kineticAcqs (list): acquisitions used for kinetics computation
        emgAcqs (list): acquisitions used for emg computation
    """

    def __init__(self,spatioTemporalAcqs=None,kinematicAcqs=None,kineticAcqs=None,emgAcqs=None):

        super(GaitCyclesBuilder, self).__init__(
            spatioTemporalAcqs = spatioTemporalAcqs,
            kinematicAcqs=kinematicAcqs,
            kineticAcqs = kineticAcqs,
            emgAcqs = emgAcqs,
            )


    def getSpatioTemporal(self):
        """get the list of Cycle used for  spatio-temporal parameter computation
        """

        if self.spatioTemporalAcqs is not None:
            spatioTemporalCycles=list()
            for acq in  self.spatioTemporalAcqs:

                context = "Left"
                left_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())


                for i in range(0, len(left_fs_frames)-1):
                    spatioTemporalCycles.append (GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                   context))

                context = "Right"
                right_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        right_fs_frames.append(ev.GetFrame())


                for i in range(0, len(right_fs_frames)-1):
                    spatioTemporalCycles.append (GaitCycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                   context))

            return spatioTemporalCycles
        else:
            return None

    def getKinematics(self):
        """get the list of Cycle used for kinematic computation
        """

        if self.kinematicAcqs is not None:
            kinematicCycles=list()
            for acq in  self.kinematicAcqs:

                context = "Left"
                left_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())

                for i in range(0, len(left_fs_frames)-1):
                    kinematicCycles.append (GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                   context))

                context = "Right"
                right_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        right_fs_frames.append(ev.GetFrame())



                for i in range(0, len(right_fs_frames)-1):
                    kinematicCycles.append (GaitCycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                   context))

            return kinematicCycles
        else:
            return None

    def getKinetics(self):
        """get the list of Cycle used for kinetic computation
        """

        if self.kineticAcqs is not None:

            detectionTimeOffset = 0.02

            kineticCycles=list()
            for acq in  self.kineticAcqs:

                flag_kinetics,frames,frames_left,frames_right = btkTools.isKineticFlag(acq)

                if flag_kinetics:
                    context = "Left"
                    count_L=0
                    left_fs_frames=list()
                    for ev in btk.Iterate(acq.GetEvents()):
                        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                            left_fs_frames.append(ev.GetFrame())


                    for i in range(0, len(left_fs_frames)-1):
                        init =  left_fs_frames[i]
                        end =  left_fs_frames[i+1]

                        for frameKinetic in frames_left:

                            if frameKinetic<=end and frameKinetic>=init:
                                LOGGER.logger.debug("Left kinetic cycle found from %.2f to %.2f" %(left_fs_frames[i], left_fs_frames[i+1]))
                                kineticCycles.append (GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                           context))

                                count_L+=1
                    LOGGER.logger.debug("%i Left Kinetic cycles available" %(count_L))



                    context = "Right"
                    count_R=0
                    right_fs_frames=list()
                    for ev in btk.Iterate(acq.GetEvents()):
                        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                            right_fs_frames.append(ev.GetFrame())

                    for i in range(0, len(right_fs_frames)-1):
                        init =  right_fs_frames[i]
                        end =  right_fs_frames[i+1]

                        for frameKinetic in frames_right:
                            if frameKinetic<=end and frameKinetic>=init:
                                LOGGER.logger.debug("Right kinetic cycle found from %.2f to %.2f" %(right_fs_frames[i], right_fs_frames[i+1]))
                                kineticCycles.append (GaitCycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                           context))
                                count_R+=1
                    LOGGER.logger.debug("%i Right Kinetic cycles available" %(count_R))

            return kineticCycles
        else:
            return None

    def getEmg(self):
        """get the list of Cycle used for emg computation
        """

        if self.emgAcqs is not None:
            emgCycles=list()
            for acq in  self.emgAcqs:

                context = "Left"
                left_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())

                for i in range(0, len(left_fs_frames)-1):
                    emgCycles.append (GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                   context))

                context = "Right"
                right_fs_frames=list()
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        right_fs_frames.append(ev.GetFrame())

                for i in range(0, len(right_fs_frames)-1):
                    emgCycles.append (GaitCycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                   context))

            return emgCycles
        else:
            return None
