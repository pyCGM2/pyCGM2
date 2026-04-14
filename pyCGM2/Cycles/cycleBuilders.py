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

from pyCGM2.Cycles import cycleFilter

import btk

from pyCGM2.Tools import btkTools

from typing import List, Tuple, Dict, Optional, Union, Callable, Sequence


ArrayLike = Union[np.ndarray, Sequence[float], None]

def build_cycles_fromEvents(
        ipsi_fs: ArrayLike,
        ipsi_fo: ArrayLike = None,
        contra_fs: ArrayLike = None,
        contra_fo: ArrayLike = None,
        check_order_for_complete: bool = True,
    ) -> List[Dict[str, object]]:
    """
    Returns a list of dicts with mandatory keys:
      - start
      - end
      - type : "cycle" | "partial_gaitCycle" | "complete_gaitCycle"

    Optional keys (added only if present in window):
      - footOff
      - controlateral_footStrike
      - controlateral_footOff

    If check_order_for_complete=True, complete cycles must satisfy:
      start < controlateral_footOff < controlateral_footStrike < footOff < end
    otherwise they are downgraded to "partial_gaitCycle" (kept, but not 'complete').
    """
    def _as_float_array(x: ArrayLike) -> np.ndarray:
        if x is None:
            return np.asarray([], dtype=float)
        return np.asarray(x, dtype=float).reshape(-1)


    def _first_in_window(events: np.ndarray, t0: float, t1: float) -> Optional[float]:
        if events.size == 0:
            return None
        cand = events[(events > t0) & (events < t1)]
        return float(cand[0]) if cand.size else None


    ipsi_fs_arr = _as_float_array(ipsi_fs)
    if ipsi_fs_arr.size < 2:
        return []

    ipsi_fo_arr   = _as_float_array(ipsi_fo)
    contra_fs_arr = _as_float_array(contra_fs)
    contra_fo_arr = _as_float_array(contra_fo)

    out: List[Dict[str, object]] = []

    for i in range(len(ipsi_fs_arr) - 1):
        start = float(ipsi_fs_arr[i])
        end   = float(ipsi_fs_arr[i + 1])

        footOff = _first_in_window(ipsi_fo_arr, start, end)
        cFS     = _first_in_window(contra_fs_arr, start, end)
        cFO     = _first_in_window(contra_fo_arr, start, end)

        d: Dict[str, object] = {"start": start, "end": end}

        # always start from minimal classification
        cycle_type = "cycle"

        if footOff is not None:
            d["footOff"] = footOff
            cycle_type = "partial_gaitCycle"

        # complete only if all three exist
        if (footOff is not None) and (cFS is not None) and (cFO is not None):
            d["controlateral_footStrike"] = cFS
            d["controlateral_footOff"] = cFO

            if check_order_for_complete:
                ok = (start < cFO < cFS < footOff < end)
                if ok:
                    cycle_type = "complete_gaitCycle"
                else:
                    # keep it, but don't call it complete
                    cycle_type = "partial_gaitCycle"
                    d["order_issue"] = True
            else:
                cycle_type = "complete_gaitCycle"
        else:
            # if one of controlat events exists but not both, you can still store it
            if cFS is not None:
                d["controlateral_footStrike"] = cFS
            if cFO is not None:
                d["controlateral_footOff"] = cFO

        d["type"] = cycle_type
        out.append(d)

    return out





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

    



