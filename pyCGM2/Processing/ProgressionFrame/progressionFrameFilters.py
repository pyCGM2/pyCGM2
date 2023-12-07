"""
This module is dedicated to determining the progression frame of a gait analysis trial. 
It includes a filter that utilizes various procedures to identify the primary axis of 
progression and the direction (forward or backward) of the movement within the trial. 
This is essential for accurately analyzing gait patterns and kinematics in clinical and 
sports settings.

"""

import btk

import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Processing.ProgressionFrame.progressionFrameProcedures import ProgressionFrameProcedure
from typing import List, Tuple, Dict, Optional,Union,Any
class ProgressionFrameFilter(object):
    """A filter to determine the progression frame of a gait analysis trial.

    This filter applies a specified procedure to an acquisition to identify the primary 
    axis of progression and the direction of movement. It is crucial for understanding 
    the spatial orientation of the subject during the trial.

    Args:
        acq (btk.btkAcquisition): An acquisition instance containing gait data.
        progressionProcedure (ProgressionFrameProcedure): An instance of a progression frame procedure.
    """

    def __init__(self, acq:btk.btkAcquisition,progressionProcedure:ProgressionFrameProcedure):

        self.m_procedure = progressionProcedure
        self.m_acq = acq

        self.outputs = {"progressionAxis": None, "forwardProgression": None, "globalFrame": None}


    def compute(self):
        """
        Executes the progression frame computation.

        This method runs the specified progression frame procedure on the acquisition data 
        and updates the outputs with the progression axis, forward progression flag, 
        and global frame information.
        """
        progressionAxis,forwardProgression,globalFrame= self.m_procedure.compute(self.m_acq)

        self.outputs["progressionAxis"] = progressionAxis
        self.outputs["forwardProgression"] = forwardProgression
        self.outputs["globalFrame"] = globalFrame
