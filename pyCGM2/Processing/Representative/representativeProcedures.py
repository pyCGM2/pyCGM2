"""
This module provides functionality for detecting representative cycles in gait analysis. 
It includes procedures for evaluating and selecting cycles that best represent the 
typical gait pattern of an individual based on specific criteria. The primary use of 
these procedures is in clinical gait analysis where a single representative cycle is 
often required for detailed analysis.
"""

import numpy as np
import pandas as pd

import pyCGM2
LOGGER = pyCGM2.LOGGER

from pyCGM2.Processing.analysis import Analysis
from typing import List, Tuple, Dict, Optional,Union,Any


class RepresentativeProcedure(object):
    """
    A base class for procedures to identify representative cycles in gait analysis.

    This class serves as a foundation for implementing various methods of identifying 
    representative cycles from a collection of gait cycles. It provides a common interface 
    for all derived classes.
    """
    def __init__(self):
        pass



class Sangeux2015Procedure(RepresentativeProcedure):
    """
    Implementation of the representative cycle detection method as described by Sangeux in 2015.

    This procedure identifies the most representative gait cycle based on the method 
    described in Sangeux's 2015 publication. It computes the frame-by-frame median deviation 
    for specified kinematic outputs and selects the cycle with the smallest deviation.

    Reference:
        Sangeux, M. (2015). A simple method to choose the most representative stride and detect outliers.


    """

    def __init__(self):
        super(Sangeux2015Procedure, self).__init__()

        self.m_data = {}
        self.m_data["Left"] = []
        self.m_data["Right"] = []


    def setDefaultData(self):
        """
        Sets the default data for the procedure according to Sangeux's 2015 method.

        The default kinematic model outputs and their respective axis indexes are set for 
        both left and right contexts.
        """

        self.setData('Left',"LPelvisAngles",[0,1,2])
        self.setData('Left',"LHipAngles",[0,1,2])
        self.setData('Left',"LKneeAngles",[0,1])
        self.setData('Left',"LAnkleAngles",[0])
        self.setData('Left',"LFootProgressAngles",[2])

        self.setData('Right',"RPelvisAngles",[0,1,2])
        self.setData('Right',"RHipAngles",[0,1,2])
        self.setData('Right',"RKneeAngles",[0,1])
        self.setData('Right',"RAnkleAngles",[0])
        self.setData('Right',"RFootProgressAngles",[2])



    def setData(self, EventContext: str, Label: str, indexes: List[int]):
        """
        Populates the data for the procedure.

        Args:
            EventContext (str): The event context (e.g., 'Left' or 'Right').
            Label (str): The kinematic model output label.
            indexes (List[int]): The axis indexes to consider.

        Example:
            ```python
            proc = Sangeux2015Procedure()
            proc.setData("Left", "LHipAngles", [0, 2]) # 0: flexion and 2: transverse rotation
            ```
        """
        self.m_data[EventContext].append([Label,indexes])


    def _calculateFmd(self, medianeValues: np.ndarray, values: np.ndarray):
        """
        Calculates the Frame-by-Frame Median Deviation (FMD).

        Args:
            medianeValues (np.ndarray): The median values.
            values (np.ndarray): The actual values for a specific cycle.

        Returns:
            float: The calculated FMD value.
        """
        return np.divide( np.sum( np.abs(values[1:100]-medianeValues[1:100]))+
                    0.5 *( np.abs(values[0]-medianeValues[0]) +
                            np.abs(values[100]-medianeValues[100])), 100)


    def _run(self, analysis: Analysis):
        """
        Runs the Sangeux 2015 procedure on the provided analysis data.

        Computes the most representative cycle index for each event context (Left and Right) 
        based on the frame-by-frame median deviation.

        Args:
            analysis (Analysis): The analysis instance containing the gait cycles.

        Returns:
            Dict[str, int]: A dictionary with the most representative cycle index for each event context.
        """

        out={}
        for eventContext in self.m_data:

            fmds=[]
            for data in self.m_data[eventContext]:
                label = data[0]
                axes  = data[1]

                mediane = analysis.kinematicStats.data[label,eventContext]["median"]
                stridesValues = analysis.kinematicStats.data[label,eventContext]["values"]
                nStrides = len(stridesValues)

                for axis in axes:
                    fmd_byStride = [label,axis]
                    for strideIndex in range(0,nStrides):
                        fmd_byStride.append(self._calculateFmd(mediane[:,axis],stridesValues[strideIndex][:,axis]))
                    fmds.append(fmd_byStride)


            colnames = ['Label', 'Axis'] +[str(i) for i in range(0,nStrides)]
            df = pd.DataFrame(fmds, columns=colnames)

            sortedDf = df.iloc[:,2:].mean().sort_values()

            sortedStrideIndex =  [int(i) for i in sortedDf.index]

            out[eventContext] = sortedStrideIndex[0]

        return out
