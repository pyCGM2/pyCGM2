# -*- coding: utf-8 -*-
import pyCGM2; LOGGER = pyCGM2.LOGGER
import numpy as np
import pandas as pd
from collections import OrderedDict

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
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
            try:
                proxLabel = self.scoreProcedure.m_scoreDefinition[nodeLabel]["proximal"]
                distLabel = self.scoreProcedure.m_scoreDefinition[nodeLabel]["distal"]

                proxNode = self.m_model.getSegment(proxLabel).getReferential('TF').getNodeTrajectory(nodeLabel)
                distNode = self.m_model.getSegment(distLabel).getReferential('TF').getNodeTrajectory(nodeLabel)
                score = numeric.rms((proxNode-distNode),axis = 1)

                scoreValues = np.array([score, np.zeros(self.acq.GetPointFrameNumber()), np.zeros(self.acq.GetPointFrameNumber())]).T

                btkTools.smartAppendPoint(self.acq, str(nodeLabel+"_Score"),scoreValues, PointType=btk.btkPoint.Scalar,desc="Score")
            except:
                LOGGER.logger.error("[pyCGM2] Score residual for node (%s) not computed"%(nodeLabel))

    def getStats(self,ipp,jointLabels,EventContext="Overall",df=None):

        series = list()

        if EventContext == "Overall":
            for jointLabel in jointLabels:

                score = np.concatenate([self.acq.GetPoint("L"+jointLabel+"_Score").GetValues()[:,0],self.acq.GetPoint("R"+jointLabel+"_Score").GetValues()[:,0]])


                iDict = OrderedDict([('Ipp', ipp),
                             ('JointLabel', jointLabel),
                             ('EventContext', EventContext),
                             ('Mean', score.mean()),
                             ('Std', score.std()),
                             ('Max', score.max()),
                             ('Min', score.min())])
                serie = pd.Series(iDict)
                series.append(serie)

        if df is not None:
            return pd.concat([df,pd.DataFrame(series)])
        else:
            return pd.DataFrame(series)
