# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

"""
This module aims to detect the progression frame of a trial.

The  filter `ProgressionFrameFilter` calls a  procedure, and
return the progression axis and a flag indicating the forward/backward progression.

"""
import numpy as np

from pyCGM2.Tools import  btkTools
import pyCGM2; LOGGER = pyCGM2.LOGGER

class ProgressionFrameFilter(object):
    """Progression Filter

    Args:
        acq (btk.Acquisition): an acquisition
        progressionProcedure (pyCGM2.Processing.ProgressionFrame.progressionFrameProcedures.ProgressionFrameProcedure): a procedure instance

    """

    def __init__(self, acq,progressionProcedure):

        self.m_procedure = progressionProcedure
        self.m_acq = acq

        self.outputs = {"progressionAxis": None, "forwardProgression": None, "globalFrame": None}


    def compute(self):
        """ run the filter"""
        progressionAxis,forwardProgression,globalFrame= self.m_procedure.compute(self.m_acq)

        self.outputs["progressionAxis"] = progressionAxis
        self.outputs["forwardProgression"] = forwardProgression
        self.outputs["globalFrame"] = globalFrame
