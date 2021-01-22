# -*- coding: utf-8 -*-
import logging
import numpy as np

try: 
    from pyCGM2 import btk
except:
    logging.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk


from  pyCGM2.Math import numeric
from pyCGM2.Tools import  btkTools



class GeneralScoreResidualProcedure(object):

    def __init__(self,model):

        self.m_scoreDefinition = dict()
        self.model = model

    def setDefinition(self, nodeLabel, proxSegLabel, distSegLabel):
        self.m_scoreDefinition[nodeLabel] = {"proximal": proxSegLabel , "distal": distSegLabel }


class ModelScoreResidualProcedure(object):

    def __init__(self, model):

        self.model = model
        self.m_scoreDefinition = dict()

        for it in self.model.m_jointCollection:
            self.m_scoreDefinition[it.m_nodeLabel] = {"proximal": it.m_proximalLabel , "distal": it.m_distalLabel }



class ScoreResidualFilter(object):

    def __init__(self,acq, scoreProcedure):
        self.scoreProcedure = scoreProcedure
        self.acq = acq

        self.m_model = scoreProcedure.model

    def compute(self):

        for nodeLabel in self.scoreProcedure.m_scoreDefinition.keys():

            proxLabel = self.scoreProcedure.m_scoreDefinition[nodeLabel]["proximal"]
            distLabel = self.scoreProcedure.m_scoreDefinition[nodeLabel]["distal"]

            proxNode =  self.m_model.getSegment(proxLabel).anatomicalFrame.getNodeTrajectory(nodeLabel)
            distNode = self.m_model.getSegment(distLabel).anatomicalFrame.getNodeTrajectory(nodeLabel)
            score = numeric.rms((proxNode-distNode),axis = 1)

            scoreValues = np.array([score, np.zeros(self.acq.GetPointFrameNumber()), np.zeros(self.acq.GetPointFrameNumber())]).T

            btkTools.smartAppendPoint(self.acq, str(nodeLabel+"_Score"),scoreValues, PointType=btk.btkPoint.Scalar,desc="Score")
