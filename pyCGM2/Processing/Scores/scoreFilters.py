# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

"""
This module is dedicated to the quantification of a Score, i e a Global index
caracterizing the movement performed

The  filter `ScoreFilter` calls a specific procedure, and return scores values.

"""
import numpy as np
from pyCGM2.Math import numeric
import pyCGM2
LOGGER = pyCGM2.LOGGER


class ScoreFilter(object):
    """the score filter

    Args:
        scoreProcedure (pyCGM2.processing.Scores.scoreProcedures.ScoreProcedure): a  procedure instance
        analysis (pyCGM2.Processing.analysis.Analysis): and `analysis` instance
        normativeDataSet (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance

    """


    def __init__(self, scoreProcedure, analysis, normativeDataSet):

        self.m_score = scoreProcedure

        # construct normative data
        self.m_normativeData =  normativeDataSet.data

        self.m_analysis=analysis


    def compute(self):


        if isinstance(self.m_score,pyCGM2.Processing.Scores.scoreProcedures.ScoreProcedure):
            descriptiveGvsStats,descriptiveGpsStats_context,descriptiveGpsStats = self.m_score._compute(self.m_analysis,self.m_normativeData)
            self.m_analysis.setGps(descriptiveGpsStats,descriptiveGpsStats_context)
            self.m_analysis.setGvs(descriptiveGvsStats)
        else:
            raise Exception("[pyCGM2] - the loaded procedure is not a ScoreProcedure instance")
