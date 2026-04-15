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
from pyCGM2.Cycles import cycleFilter
import btk



from typing import List, Tuple, Dict, Optional, Union, Callable, Sequence



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

    def __init__(self,acqs:List[btk.btkAcquisition]):
        self.acqs =acqs

    def getCycles(self):
        """
        Get the list of Cycles used for spatio-temporal parameter computation.

        Returns:
            Optional[List[Cycle]]: List of cycles for spatio-temporal parameter computation.
        """

        if self.acqs is not None:
            cycles=[]
            for acq in  self.acqs:

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
                    cycles.append (cycleFilter.Cycle(acq, startFrame,endFrame,"Left"))
                    cycles.append (cycleFilter.Cycle(acq, startFrame,endFrame,"Right"))
                    LOGGER.logger.info("[pyCGM2] left and Right context - time normalization from time boudaries")

                if len(left_fs_frames) >1:
                    for i in range(0, len(left_fs_frames)-1):
                        cycles.append (cycleFilter.Cycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                       "Left"))
                elif len(left_fs_frames) ==1:
                    LOGGER.logger.warning("[pyCGM2] No left cycles, only one left foot strike detected)")
                else:
                    LOGGER.logger.warning("[pyCGM2] No left cycles")

                if len(right_fs_frames)>1:
                    for i in range(0, len(right_fs_frames)-1):
                        cycles.append (Cycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                       "Right"))
                elif len(right_fs_frames) ==1:
                    LOGGER.logger.warning("[pyCGM2] No right cycles, only one right foot strike detected)")
                else:
                    LOGGER.logger.warning("[pyCGM2] No Right cycles")

            return cycles
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

    def __init__(self,acqs:List[btk.btkAcquisition]):

        super(GaitCyclesBuilder, self).__init__(acqs)

    def getCycles(self):
        """
        Get the list of Gait Cycles used for spatio-temporal parameter computation.

        Returns:
            Optional[List[GaitCycle]]: List of cycles for spatio-temporal parameter computation.
        """

        if self.acqs is not None:
            cycles=[]
            for acq in  self.acqs:

                context = "Left"
                left_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        left_fs_frames.append(ev.GetFrame())


                for i in range(0, len(left_fs_frames)-1):
                    cycles.append (cycleFilter.GaitCycle(acq, left_fs_frames[i],left_fs_frames[i+1],
                                                   context))

                context = "Right"
                right_fs_frames=[]
                for ev in btk.Iterate(acq.GetEvents()):
                    if ev.GetContext() == context and ev.GetLabel() == "Foot Strike":
                        right_fs_frames.append(ev.GetFrame())


                for i in range(0, len(right_fs_frames)-1):
                    cycles.append (cycleFilter.GaitCycle(acq, right_fs_frames[i],right_fs_frames[i+1],
                                                   context))

            return cycles
        else:
            return None

    



