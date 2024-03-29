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
            return self.acq.GetPoint(pointLabel).GetValues()[self.begin-self.firstFrame:self.end-self.firstFrame+1,0:3] # 0.3 because openma::Ts includes a forth column (i.e residual)
        else:
            LOGGER.logger.debug("[pyCGM2] the point Label %s doesn t exist " % (pointLabel))
            return None



    def getPointTimeSequenceDataNormalized(self,pointLabel:str):
        """
        Get Time-normalized a point label.

        Args:
            pointLabel (str): Point label.

        Returns:
            np.ndarray: Time-normalized data of the specified point label.
        """

        data = self.getPointTimeSequenceData(pointLabel)
        if data is None:
            out=np.zeros((101,3))
        else:
            out = MathNormalisation.timeSequenceNormalisation(101,data)

        return out

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


    def getAnalogTimeSequenceDataNormalized(self,analogLabel:str):
        """
        Get time-normalized analog data of the cycle.

        Args:
            analogLabel (str): Analog label.

        Returns:
            Optional[np.ndarray]: Analog data of the specified label.
        """

        data = self.getAnalogTimeSequenceData(analogLabel)
        if data is None:
            out=np.zeros((101,3))
        else:
            out = MathNormalisation.timeSequenceNormalisation(101,data)

        return  out

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


# ----- PATTERN BUILDER -----





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
        self.muscleGeometryCycles = None
        self.muscleDynamicCycles = None


    def setSpatioTemporalCycles(self,spatioTemporalCycles_instance):
        self.spatioTemporalCycles = spatioTemporalCycles_instance

    def setKinematicCycles(self,kinematicCycles_instance):
        self.kinematicCycles = kinematicCycles_instance

    def setKineticCycles(self,kineticCycles_instance):
        self.kineticCycles = kineticCycles_instance

    def setEmgCycles(self,emgCycles_instance):
        self.emgCycles = emgCycles_instance

    def setMuscleGeometryCycles(self,muscleGeometryCycles_instance):
        self.muscleGeometryCycles = muscleGeometryCycles_instance

    def setMuscleDynamicCycles(self,muscleDynamicCycles_instance):
        self.muscleDynamicCycles = muscleDynamicCycles_instance

# --- BUILDER
class CyclesBuilder(object):
    """
    Builder of generic cycles.

    Args:
        spatioTemporalAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for spatio-temporal parameter computation.
        kinematicAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for kinematics computation.
        kineticAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for kinetics computation.
        emgAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for EMG computation.
        muscleGeometryAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for muscle geometry computation.
        muscleDynamicAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for muscle dynamic computation.
    """

    def __init__(self,spatioTemporalAcqs:Optional[List[btk.btkAcquisition]]=None,
                 kinematicAcqs:Optional[List[btk.btkAcquisition]]=None,
                 kineticAcqs:Optional[List[btk.btkAcquisition]]=None,
                 emgAcqs:Optional[List[btk.btkAcquisition]]=None,
                 muscleGeometryAcqs:Optional[List[btk.btkAcquisition]]=None,
                 muscleDynamicAcqs:Optional[List[btk.btkAcquisition]]=None):

        self.spatioTemporalAcqs =spatioTemporalAcqs
        self.kinematicAcqs =kinematicAcqs
        self.kineticAcqs =kineticAcqs
        self.emgAcqs =emgAcqs
        self.muscleGeometryAcqs = muscleGeometryAcqs
        self.muscleDynamicAcqs = muscleDynamicAcqs

    def getSpatioTemporal(self):
        """
        Get the list of Cycles used for spatio-temporal parameter computation.

        Returns:
            Optional[List[Cycle]]: List of cycles for spatio-temporal parameter computation.
        """

        if self.spatioTemporalAcqs is not None:
            spatioTemporalCycles=[]
            for acq in  self.spatioTemporalAcqs:

                startFrame = acq.GetFirstFrame()
                endFrame = acq.GetLastFrame()

                context = "Left"
                left_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())
                #
                # for ev in  acq.findChild(ma.T_Node,"SortedEvents").findChildren(ma.T_Event,"Foot Strike",[["context","Left"]]):
                #     left_fs_frames.append(ev.time())

                context = "Right"
                right_fs_frames=[]
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
        """
        Get the list of Cycles used for kinematic computation.

        Returns:
            Optional[List[Cycle]]: List of cycles for kinematic computation.
        """
        if self.kinematicAcqs is not None:
            kinematicCycles=[]
            for acq in  self.kinematicAcqs:

                startFrame = acq.GetFirstFrame()
                endFrame = acq.GetLastFrame()

                context = "Left"
                left_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())


                context = "Right"
                right_fs_frames=[]
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
        """
        Get the list of Cycles used for kinetic computation.

        Returns:
            Optional[List[Cycle]]: List of cycles for kinetic computation.
        """
        if self.kineticAcqs is not None:

            detectionTimeOffset = 0.02

            kineticCycles=[]
            for acq in  self.kineticAcqs:

                flag_kinetics,frames,frames_left,frames_right = btkTools.isKineticFlag(acq)

                startFrame = acq.GetFirstFrame()
                endFrame = acq.GetLastFrame()

                if flag_kinetics:

                    context = "Left"
                    left_fs_frames=[]
                    for ev in btk.Iterate(acq.GetEvents()):
                        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                            left_fs_frames.append(ev.GetFrame())

                    context = "Right"
                    right_fs_frames=[]
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
        """
        Get the list of Cycles used for EMG computation.

        Returns:
            Optional[List[Cycle]]: List of cycles for EMG computation.
        """
        if self.emgAcqs is not None:
            emgCycles=[]
            for acq in  self.emgAcqs:

                startFrame = acq.GetFirstFrame()
                endFrame = acq.GetLastFrame()

                context = "Left"
                left_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())

                context = "Right"
                right_fs_frames=[]
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
    
    def getMuscleGeometry(self):
        """
        Get the list of Cycles used for muscle geometry computation.

        Returns:
            Optional[List[Cycle]]: List of cycles for muscle geometry computation.
        """
        if self.muscleGeometryAcqs is not None:
            muscleGeometryCycles=[]
            for acq in  self.muscleGeometryAcqs:

                startFrame = acq.GetFirstFrame()
                endFrame = acq.GetLastFrame()

                context = "Left"
                left_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())


                context = "Right"
                right_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        right_fs_frames.append(ev.GetFrame())

                if left_fs_frames == [] and  right_fs_frames == []:
                    muscleGeometryCycles.append (Cycle(acq, startFrame,endFrame,"Left"))
                    muscleGeometryCycles.append (Cycle(acq, startFrame,endFrame,"Right"))
                    LOGGER.logger.info("[pyCGM2] left and Right context - time normalization from time boudaries")

                if len(left_fs_frames) >1:
                    for i in range(0, len(left_fs_frames)-1):
                        muscleGeometryCycles.append (Cycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                       "Left"))
                elif len(left_fs_frames) ==1:
                    LOGGER.logger.warning("[pyCGM2] No left cycles, only one left foot strike detected)")
                else:
                    LOGGER.logger.warning("[pyCGM2] No left cycles")

                if len(right_fs_frames) >1:
                    for i in range(0, len(right_fs_frames)-1):
                        muscleGeometryCycles.append (Cycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                       "Right"))
                elif len(right_fs_frames) ==1:
                    LOGGER.logger.warning("[pyCGM2] No right cycles, only one right foot strike detected)")
                else:
                    LOGGER.logger.warning("[pyCGM2] No Right cycles")

            return muscleGeometryCycles
        else:
            return None

    def getMuscleDynamic(self):
        """
        Get the list of Cycles used for muscle dynamic computation.

        Returns:
            Optional[List[Cycle]]: List of cycles for muscle dynamic computation.
        """
        if self.muscleDynamicAcqs is not None:

            detectionTimeOffset = 0.02

            muscleDynamicCycles=[]
            for acq in  self.muscleDynamicAcqs:

                flag_kinetics,frames,frames_left,frames_right = btkTools.isKineticFlag(acq)

                startFrame = acq.GetFirstFrame()
                endFrame = acq.GetLastFrame()

                if flag_kinetics:

                    context = "Left"
                    left_fs_frames=[]
                    for ev in btk.Iterate(acq.GetEvents()):
                        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                            left_fs_frames.append(ev.GetFrame())

                    context = "Right"
                    right_fs_frames=[]
                    for ev in btk.Iterate(acq.GetEvents()):
                        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                            right_fs_frames.append(ev.GetFrame())


                    if left_fs_frames == [] and right_fs_frames==[]:
                        muscleDynamicCycles.append (Cycle(acq, startFrame,endFrame,"Left"))
                        muscleDynamicCycles.append (Cycle(acq, startFrame,endFrame,"Right"))
                        LOGGER.logger.info("[pyCGM2] left - time normalization from time boudaries")


                    count_L=0
                    if len(left_fs_frames)>1:
                        for i in range(0, len(left_fs_frames)-1):
                            init =  left_fs_frames[i]
                            end =  left_fs_frames[i+1]

                            for frameKinetic in frames_left:

                                if frameKinetic<=end and frameKinetic>=init:
                                    LOGGER.logger.debug("Left kinetic cycle found from %.2f to %.2f" %(left_fs_frames[i], left_fs_frames[i+1]))
                                    muscleDynamicCycles.append (Cycle(acq, left_fs_frames[i],left_fs_frames[i+1],
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
                                    muscleDynamicCycles.append (Cycle(acq, right_fs_frames[i],right_fs_frames[i+1],
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

class GaitCyclesBuilder(CyclesBuilder):
    """
        Builder of gait cycles.

        Args:
            spatioTemporalAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for spatio-temporal parameter computation.
            kinematicAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for kinematics computation.
            kineticAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for kinetics computation.
            emgAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for EMG computation.
            muscleGeometryAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for muscle geometry computation.
            muscleDynamicAcqs (Optional[List[btk.btkAcquisition]]): Acquisitions used for muscle dynamic computation.
        """

    def __init__(self,spatioTemporalAcqs:Optional[List[btk.btkAcquisition]]=None,
                 kinematicAcqs:Optional[List[btk.btkAcquisition]]=None,
                 kineticAcqs:Optional[List[btk.btkAcquisition]]=None,
                 emgAcqs:Optional[List[btk.btkAcquisition]]=None,
                 muscleGeometryAcqs:Optional[List[btk.btkAcquisition]]=None,
                 muscleDynamicAcqs:Optional[List[btk.btkAcquisition]]=None):

        super(GaitCyclesBuilder, self).__init__(
            spatioTemporalAcqs = spatioTemporalAcqs,
            kinematicAcqs=kinematicAcqs,
            kineticAcqs = kineticAcqs,
            emgAcqs = emgAcqs,
            muscleGeometryAcqs = muscleGeometryAcqs,
            muscleDynamicAcqs = muscleDynamicAcqs
            )

    def getSpatioTemporal(self):
        """
        Get the list of Gait Cycles used for spatio-temporal parameter computation.

        Returns:
            Optional[List[GaitCycle]]: List of cycles for spatio-temporal parameter computation.
        """

        if self.spatioTemporalAcqs is not None:
            spatioTemporalCycles=[]
            for acq in  self.spatioTemporalAcqs:

                context = "Left"
                left_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())


                for i in range(0, len(left_fs_frames)-1):
                    spatioTemporalCycles.append (GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                   context))

                context = "Right"
                right_fs_frames=[]
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
        """
        Get the list of Gait Cycles used for kinematic computation.

        Returns:
            Optional[List[GaitCycle]]: List of cycles for kinematic computation.
        """

        if self.kinematicAcqs is not None:
            kinematicCycles=[]
            for acq in  self.kinematicAcqs:

                context = "Left"
                left_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())

                for i in range(0, len(left_fs_frames)-1):
                    kinematicCycles.append (GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                   context))

                context = "Right"
                right_fs_frames=[]
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
        """
        Get the list of Gait Cycles used for kinetic computation.

        Returns:
            Optional[List[GaitCycle]]: List of cycles for kinetic computation.
        """

        if self.kineticAcqs is not None:

            detectionTimeOffset = 0.02

            kineticCycles=[]
            for acq in  self.kineticAcqs:

                flag_kinetics,frames,frames_left,frames_right = btkTools.isKineticFlag(acq)

                if flag_kinetics:
                    context = "Left"
                    count_L=0
                    left_fs_frames=[]
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
                    right_fs_frames=[]
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
        """
        Get the list of Gait Cycles used for EMG computation.

        Returns:
            Optional[List[GaitCycle]]: List of cycles for EMG computation.
        """

        if self.emgAcqs is not None:
            emgCycles=[]
            for acq in  self.emgAcqs:

                context = "Left"
                left_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())

                for i in range(0, len(left_fs_frames)-1):
                    emgCycles.append (GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                   context))

                context = "Right"
                right_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        right_fs_frames.append(ev.GetFrame())

                for i in range(0, len(right_fs_frames)-1):
                    emgCycles.append (GaitCycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                   context))

            return emgCycles
        else:
            return None


    def getMuscleGeometry(self):
        """
        Get the list of Gait Cycles used for muscle geometry computation.

        Returns:
            Optional[List[GaitCycle]]: List of cycles for muscle geometry computation.
        """
        if self.muscleGeometryAcqs is not None:
            muscleGeometryCycles=[]
            for acq in  self.muscleGeometryAcqs:

                context = "Left"
                left_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())

                for i in range(0, len(left_fs_frames)-1):
                    muscleGeometryCycles.append (GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                   context))

                context = "Right"
                right_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        right_fs_frames.append(ev.GetFrame())



                for i in range(0, len(right_fs_frames)-1):
                    muscleGeometryCycles.append (GaitCycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                   context))

            return muscleGeometryCycles
        else:
            return None

    
    def getMuscleDynamic(self):
        """
        Get the list of Gait Cycles used for muscle dynamic computation.

        Returns:
            Optional[List[GaitCycle]]: List of cycles for muscle dynamic computation.
        """

        if self.muscleDynamicAcqs is not None:

            detectionTimeOffset = 0.02

            muscleDynamicCycles=[]
            for acq in  self.muscleDynamicAcqs:

                flag_kinetics,frames,frames_left,frames_right = btkTools.isKineticFlag(acq)

                if flag_kinetics:
                    context = "Left"
                    count_L=0
                    left_fs_frames=[]
                    for ev in btk.Iterate(acq.GetEvents()):
                        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                            left_fs_frames.append(ev.GetFrame())


                    for i in range(0, len(left_fs_frames)-1):
                        init =  left_fs_frames[i]
                        end =  left_fs_frames[i+1]

                        for frameKinetic in frames_left:

                            if frameKinetic<=end and frameKinetic>=init:
                                LOGGER.logger.debug("Left kinetic cycle found from %.2f to %.2f" %(left_fs_frames[i], left_fs_frames[i+1]))
                                muscleDynamicCycles.append (GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                           context))

                                count_L+=1
                    LOGGER.logger.debug("%i Left Kinetic cycles available" %(count_L))



                    context = "Right"
                    count_R=0
                    right_fs_frames=[]
                    for ev in btk.Iterate(acq.GetEvents()):
                        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                            right_fs_frames.append(ev.GetFrame())

                    for i in range(0, len(right_fs_frames)-1):
                        init =  right_fs_frames[i]
                        end =  right_fs_frames[i+1]

                        for frameKinetic in frames_right:
                            if frameKinetic<=end and frameKinetic>=init:
                                LOGGER.logger.debug("Right kinetic cycle found from %.2f to %.2f" %(right_fs_frames[i], right_fs_frames[i+1]))
                                muscleDynamicCycles.append (GaitCycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                           context))
                                count_R+=1
                    LOGGER.logger.debug("%i Right Kinetic cycles available" %(count_R))

            return muscleDynamicCycles
        else:
            return None

# ----- FILTER -----
class CyclesFilter:
    """ Filter buiding a `Cycles` instance.
    """

    __builder = None

    def setBuilder(self, builder:CyclesBuilder):
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

        spatioTemporalElements = self.__builder.getSpatioTemporal()
        cycles.setSpatioTemporalCycles(spatioTemporalElements)

        kinematicElements = self.__builder.getKinematics()
        cycles.setKinematicCycles(kinematicElements)

        kineticElements = self.__builder.getKinetics()
        cycles.setKineticCycles(kineticElements)

        emgElements = self.__builder.getEmg()
        cycles.setEmgCycles(emgElements)

        muscleGeometryElements = self.__builder.getMuscleGeometry()
        cycles.setMuscleGeometryCycles(muscleGeometryElements)

        muscleDynamicElements = self.__builder.getMuscleDynamic()
        cycles.setMuscleDynamicCycles(muscleDynamicElements)

        return cycles


def spatioTemporelParameter_descriptiveStats(cycles:Cycles, label:str,context:str):

    """
    Compute descriptive statistics of spatio-temporal parameters from a `cycles` instance.

    Args:
        cycles (Cycles): Cycles instance.
        label (str): Spatio-temporal label.
        context (str): Event context (e.g., Left, Right).

    Returns:
        Dict: Dictionary containing mean, std, median, and values of the spatio-temporal parameters.
    """

    outDict={}

    n=len([cycle for cycle in cycles if cycle.enableFlag and cycle.context==context])
    val = np.zeros((n))

    i=0
    for cycle in cycles:
        if cycle.enableFlag and cycle.context==context:
            val[i] = cycle.getSpatioTemporalParameter(label)
            i+=1
    outDict = {'mean':np.mean(val),'std':np.std(val),'median':np.median(val),'values': val}

    return outDict


def point_descriptiveStats(cycles: Cycles, label: str, context: str):
    """
    Compute descriptive statistics of point parameters from a `cycles` instance.

    Args:
        cycles (Cycles): Cycles instance.
        label (str): Point label.
        context (str): Event context (e.g., Left, Right).

    Returns:
        Dict: Dictionary containing mean, std, median, and values of the point parameters.
    """

    outDict={}

    n=len([cycle for cycle in cycles if cycle.enableFlag and cycle.context==context]) # list comprehension , get number of enabled cycle

    x=np.empty((101,n))
    y=np.empty((101,n))
    z=np.empty((101,n))

    listOfPointValues=[]

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



def analog_descriptiveStats(cycles: Cycles, label: str, context: str):
    """
    Compute descriptive statistics of analog parameters from a `cycles` instance.

    Args:
        cycles (Cycles): Cycles instance.
        label (str): Analog label.
        context (str): Event context (e.g., Left, Right).

    Returns:
        Dict: Dictionary containing mean, std, median, maximal values, and values of the analog parameters.
    """

    outDict={}

    n=len([cycle for cycle in cycles if cycle.enableFlag and cycle.context==context]) # list comprehension , get number of enabled cycle

    listOfPointValues=[]
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


def construcGaitCycle(acq: btk.btkAcquisition):
    """
    Construct gait cycle from an acquisition.

    Args:
        acq (btk.btkAcquisition): An acquisition instance.

    Returns:
        List: List of constructed GaitCycle instances.
    """

    gaitCycles=[]

    context = "Left"
    left_fs_frames=[]
    for ev in btk.Iterate(acq.GetEvents()):
        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
            left_fs_frames.append(ev.GetFrame())

    for i in range(0, len(left_fs_frames)-1):
        gaitCycles.append (GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                       context))

    context = "Right"
    right_fs_frames=[]
    for ev in btk.Iterate(acq.GetEvents()):
        if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
            right_fs_frames.append(ev.GetFrame())

    for i in range(0, len(right_fs_frames)-1):
        gaitCycles.append (GaitCycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                       context))

    return gaitCycles