# -*- coding: utf-8 -*-
"""
This module contains procedures for assessing the quality of the model
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

class QualityProcedure(object):
    def __init__(self):
        pass


class GeneralScoreResidualProcedure(QualityProcedure):
    """Procedure for calculating SCORE residual on given segments of the model

    The main goal of the procedure is to populate the attribute `m_scoreDefinition`
    which will be handle by the ScoreresidualFilter

    Args:
        model ( pyCGM2.Model.model.Model)

    """

    def __init__(self,model):
        super(GeneralScoreResidualProcedure, self).__init__()

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


class ModelScoreResidualProcedure(QualityProcedure):
    """Procedure for calculating SCORE residual for **all** joints of the model

    The procedure populates its attribute `m_scoreDefinition` from the geometric attributes  of the model

    Args:
        model ( pyCGM2.Model.model.Model)

    """

    def __init__(self, model):
        super(ModelScoreResidualProcedure, self).__init__()

        self.model = model
        self.m_scoreDefinition = dict()

        for it in self.model.m_jointCollection:
            self.m_scoreDefinition[it.m_nodeLabel] = {"proximal": it.m_proximalLabel , "distal": it.m_distalLabel }