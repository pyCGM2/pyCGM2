"""
This module aims to detect a representative cycle from a set of gait cycles. It is
useful in gait analysis for identifying a cycle that best represents the typical
movement pattern of an individual.

The filter `RepresentativeCycleFilter` utilizes a specified procedure to determine
the most representative cycle among a collection of cycles for both left and right
event contexts in gait analysis.


"""
import pyCGM2
LOGGER = pyCGM2.LOGGER


from pyCGM2.Processing.Representative.representativeProcedures import RepresentativeProcedure
from pyCGM2.Processing.analysis import Analysis
from typing import List, Tuple, Dict, Optional,Union,Any


class RepresentativeCycleFilter(object):
    """
    A filter for identifying the most representative cycle in gait analysis.

    This class applies a specified procedure to an analysis instance to determine
    the most representative cycle of gait. It is particularly useful in scenarios
    where a single cycle needs to be chosen as a representative for various analyses
    such as average kinematics, kinetics, or EMG patterns.

    Attributes:
        m_procedure (RepresentativeProcedure): The procedure used to determine the representative cycle.
        m_analysis (Analysis): The analysis instance containing the gait cycles.

    Args:
        analysisInstance (Analysis): An `Analysis` instance containing the gait cycles.
        representativeProcedure (RepresentativeProcedure): A procedure instance to determine the representative cycle.
    """

    def __init__(self, analysisInstance:Analysis, representativeProcedure:RepresentativeProcedure):

        self.m_procedure = representativeProcedure
        self.m_analysis = analysisInstance



    def run(self):
        """
        Executes the representative cycle selection procedure on the provided analysis data.

        This method runs the specified representative procedure on the analysis instance
        to identify the most representative cycle for both left and right event contexts.

        Returns:
            Dict[str, int]: A dictionary containing the indices of the representative cycles
                            for both left and right contexts. Keys are 'Left' and 'Right'.
        """

        representativeCycleIndex = self.m_procedure._run(self.m_analysis)

        return representativeCycleIndex
