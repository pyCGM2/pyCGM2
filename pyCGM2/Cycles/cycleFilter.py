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

from pyCGM2.Cycles import cycleBuilders
from pyCGM2.Processing.ProgressionFrame import progressionFrameFilters
from pyCGM2.Processing.ProgressionFrame import progressionFrameProcedures


import btk

from pyCGM2.Tools import btkTools

from typing import List, Tuple, Dict, Optional, Union, Callable






class Cycle(object):
    """
    Generic Cycle class constructor.

    Args:
        acq (btk.btkAcquisition): An acquisition instance.
        startFrame (int): Start frame of the cycle.
        endFrame (int): End frame of the cycle.
        context (str): Context of the cycle (e.g., 'Left', 'Right').
        enableFlag (bool): Flag to indicate if the cycle will be used in further computation. Defaults to True.
    """


    def __init__(self,acq:btk.btkAcquisition,startFrame:int,endFrame:int,context:str, enableFlag:bool = True):

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

        self.stps ={}

        LOGGER.logger.debug("cycle makes from Frame %d   to  %d   (%s) " % (self.begin, self.end, self.context))



    def setEnableFlag(self,flag:bool):
        """
        Enable or disable the cycle.

        Args:
            flag (bool): Boolean flag to enable or disable the cycle.
        """
        self.enableFlag = flag

    def addDiscreteData(self,label,value,instant):
        pass

    def getPointTimeSequenceData(self,pointLabel:str):
        """
        Get point data of the cycle.

        Args:
            pointLabel (str): Point label.

        Returns:
            Optional[np.ndarray]: Temporal data of the specified point label.
        """

        if btkTools.isPointExist(self.acq,pointLabel):
            return self.acq.GetPoint(pointLabel).GetValues()[self.begin-self.firstFrame:self.end-self.firstFrame+1,0:3] 
        else:
            LOGGER.logger.debug("[pyCGM2] the point Label %s doesn t exist " % (pointLabel))
            return None


    def getAnalogTimeSequenceData(self,analogLabel:str):
        """
        Get analog data of the cycle.

        Args:
            analogLabel (str): Analog label.

        Returns:
            Optional[np.ndarray]: Analog data of the specified label.
        """
        if btkTools.isAnalogExist(self.acq,analogLabel):
            return  self.acq.GetAnalog(analogLabel).GetValues()[int((self.begin-self.firstFrame) * self.appf) : int((self.end-self.firstFrame+1) * self.appf),:]
        else:
            LOGGER.logger.debug("[pyCGM2] the Analog Label %s doesn t exist" % (analogLabel))
            return None


    def getEvents(self,context:str="All"):
        """
        Get all events of the cycle.

        Args:
            context (str): Event context (All, Left, or Right).

        Returns:
            List[btk.Event]: List of events in the specified context.
        """
        events = []
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

    """
    GaitCycle class constructor, inherited from Cycle. Defines a gait cycle.

    Args:
        gaitAcq (btk.btkAcquisition): An acquisition instance for the gait cycle.
        startFrame (int): Start frame of the gait cycle.
        endFrame (int): End frame of the gait cycle.
        context (str): Context of the gait cycle (e.g., 'Left', 'Right').
        enableFlag (bool): Flag to indicate if the cycle will be used in further computation. Defaults to True.

    Notes:
        - By default, X0 and Y0 are the longitudinal and lateral global axes respectively.
        - `GaitCycle` construction computes spatio-temporal parameters automatically.
        - Spatio-temporal parameters include 'duration', 'cadence', 'stanceDuration', 'stepDuration', etc.
    """


    STP_LABELS=["duration","cadence",
                "stanceDuration", "stepDuration", "doubleStance1Duration",
                "doubleStance2Duration","simpleStanceDuration","stancePhase",
                "swingDuration", "swingPhase", "doubleStance1", "doubleStance2",
                "simpleStance", "stepPhase","strideLength", "stepLength",
                "strideWidth", "speed"]


    def __init__(self,gaitAcq:btk.btkAcquisition,startFrame:int,endFrame:int,context:str, enableFlag:bool = True):
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
        """
        Compute spatio-temporal parameters for the gait cycle.
        """

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

    def getSpatioTemporalParameter(self,label:str):
        """ Return a spatio-temporal parameter.

        Args:
            label (str): Label of the desired spatio-temporal parameter.

        Returns:
            float: Value of the specified spatio-temporal parameter.
        """

        return self.stps[label]


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
        self.cycles = None


    def setCycles(self,cycles):
        self.cycles = cycles






    

# ----- FILTER -----
class CyclesFilter:
    """ Filter buiding a `Cycles` instance.
    """

    __builder = None

    def setBuilder(self, builder):
        """Set the builder

        Args:
            builder (CyclesBuilder): a concrete cycle builder

        """
        self.__builder = builder


    def build(self):
        """
        Build and return a `Cycles` instance using the set builder.

        Returns:
            Cycles: An instance of `Cycles` constructed using the current builder.
        """
        cycles = Cycles()

        cyclesCollection = self.__builder.getCycles()
        cycles.setCycles(cyclesCollection)

        

        return cycles


