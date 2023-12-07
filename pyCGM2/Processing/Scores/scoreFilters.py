"""
This module is dedicated to the quantification of a Score, i.e., a global index
characterizing the movement performed.

The filter `ScoreFilter` calls a specific procedure and returns score values.
"""
import pyCGM2
LOGGER = pyCGM2.LOGGER

from pyCGM2.Processing.Scores.scoreProcedures import ScoreProcedure
from pyCGM2.Processing.analysis import Analysis
from pyCGM2.Report.normativeDatasets import NormativeData
from typing import List, Tuple, Dict, Optional,Union,Any


class ScoreFilter(object):
    """
    A filter for calculating scores using a specified scoring procedure.

    This filter takes an analysis instance and a normative data set, and applies
    a scoring procedure to compute scores like Gait Profile Score (GPS).

    Args:
        scoreProcedure (ScoreProcedure): A procedure instance for calculating the score.
        analysis (Analysis): An `Analysis` instance containing the data to be scored.
        normativeDataSet (NormativeData): A `NormativeData` instance for normative comparison.
    """


    def __init__(self, scoreProcedure:ScoreProcedure, analysis:Analysis, normativeDataSet:NormativeData):

        self.m_score = scoreProcedure

        # construct normative data
        self.m_normativeData =  normativeDataSet.data

        self.m_analysis=analysis


    def compute(self):
        """
        Computes the scores using the specified scoring procedure.

        This method applies the scoring procedure to the analysis data, comparing
        it against the normative data set, and updates the analysis instance with the computed scores.

        Raises:
            Exception: If the provided scoreProcedure is not an instance of ScoreProcedure.
        """

        if isinstance(self.m_score,pyCGM2.Processing.Scores.scoreProcedures.ScoreProcedure):
            descriptiveGvsStats,descriptiveGpsStats_context,descriptiveGpsStats = self.m_score._compute(self.m_analysis,self.m_normativeData)
            self.m_analysis.setGps(descriptiveGpsStats,descriptiveGpsStats_context)
            self.m_analysis.setGvs(descriptiveGvsStats)
        else:
            raise Exception("[pyCGM2] - the loaded procedure is not a ScoreProcedure instance")
