# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Model
#APIDOC["Draft"]=False
#--end--
"""
This module contains filters and associated procedures for assessing the quality of the model
"""

import numpy as np
import pandas as pd
from collections import OrderedDict

import pyCGM2
from  pyCGM2.Math import numeric
from pyCGM2.Tools import btkTools
LOGGER = pyCGM2.LOGGER
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")


class GeneralScoreResidualProcedure(object):
    """Procedure for calculating SCORE residual on given segments of the model

    The main goal of the procedure is to populate the attribute `m_scoreDefinition`
    which will be handle by the ScoreresidualFilter

    Args:
        model ( pyCGM2.Model.model.Model)

    """

    def __init__(self,model):

        self.m_scoreDefinition = dict()
        self.model = model

    def setDefinition(self, nodeLabel, proxSegLabel, distSegLabel):
        """set the definition dictionary.

        Args:
            nodeLabel (str): node label.
            proxSegLabel (str): proximal segment label
            distSegLabel (str): distal segment label
        """

        self.m_scoreDefinition[nodeLabel] = {"proximal": proxSegLabel , "distal": distSegLabel }


class ModelScoreResidualProcedure(object):
    """Procedure for calculating SCORE residual for **all** joints of the model

    The procedure populates its attribute `m_scoreDefinition` from the geometric attributes  of the model

    Args:
        model ( pyCGM2.Model.model.Model)

    """

    def __init__(self, model):

        self.model = model
        self.m_scoreDefinition = dict()

        for it in self.model.m_jointCollection:
            self.m_scoreDefinition[it.m_nodeLabel] = {"proximal": it.m_proximalLabel , "distal": it.m_distalLabel }



class ScoreResidualFilter(object):
    """  Calculate the SCORE residual

    Args
       acq (btk.Acquisition): an acquisition instance
       scoreProcedure (pyCGM2.Model.modelQualityFilter.(Procedure)): a score procedure instance

    **References**

      - Ehrig, Rainald M.; Heller, Markus O.; Kratzenstein, Stefan; Duda, Georg N.; Trepczynski, Adam; Taylor, William R. (2011)
      The SCoRE residual: a quality index to assess the accuracy of joint estimations.
      In : Journal of biomechanics, vol. 44, n° 7, p. 1400–1404. DOI: 10.1016/j.jbiomech.2010.12.009.

    """

    def __init__(self,acq, scoreProcedure):
        self.scoreProcedure = scoreProcedure
        self.acq = acq

        self.m_model = scoreProcedure.model

    def compute(self):
        """Run the filter

        """

        for nodeLabel in self.scoreProcedure.m_scoreDefinition.keys():
            try:
                proxLabel = self.scoreProcedure.m_scoreDefinition[nodeLabel]["proximal"]
                distLabel = self.scoreProcedure.m_scoreDefinition[nodeLabel]["distal"]

                proxNode = self.m_model.getSegment(proxLabel).getReferential('TF').getNodeTrajectory(nodeLabel)
                distNode = self.m_model.getSegment(distLabel).getReferential('TF').getNodeTrajectory(nodeLabel)
                score = numeric.rms((proxNode-distNode),axis = 1)

                scoreValues = np.array([score, np.zeros(self.acq.GetPointFrameNumber()), np.zeros(self.acq.GetPointFrameNumber())]).T

                btkTools.smartAppendPoint(self.acq, str(nodeLabel+"_Score"),scoreValues, PointType="Scalar",desc="Score")
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
