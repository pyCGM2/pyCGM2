# -*- coding: utf-8 -*-
"""
This module contains procedures for assessing the quality of the model
"""

import numpy as np
import pandas as pd
from collections import OrderedDict

import pyCGM2
from  pyCGM2.Math import geometry
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

class WandAngleQualityProcedure(QualityProcedure):
    def __init__(self):
        super(WandAngleQualityProcedure, self).__init__()

        self.m_definitions = {"LTHIplanarAngle":['LKNE', 'LHJC', "LTHI"],
                              "RTHIplanarAngle":['RKNE', 'RHJC', "RTHI"],
                              "LTIBplanarAngle":['LANK', 'LKJC', "LTIB"],
                              "RTIBplanarAngle":['RANK', 'RKJC', "RTIB"]}


    def run(self,acq):

        nrow = acq.GetPointFrameNumber()
    
        for key in self.m_definitions:
            pt1 = self.m_definitions[key][0]
            pt2 = self.m_definitions[key][1]
            pt3 = self.m_definitions[key][2]

            out = np.zeros((nrow,3))

            for i in range (0, nrow):
                pt1r = acq.GetPoint(pt1).GetResiduals()[i,0]
                pt2r = acq.GetPoint(pt2).GetResiduals()[i,0]
                pt3r = acq.GetPoint(pt3).GetResiduals()[i,0]

                gapDetected = False
                if pt1r == -1.0 or pt2r == -1.0 or pt3r == -1.0:
                    out[i,0] = 0
                    gapDetected = True

                else:
                    u1 = acq.GetPoint(pt2).GetValues()[i,:] -acq.GetPoint(pt1).GetValues()[i,:]
                    v1 = acq.GetPoint(pt3).GetValues()[i,:] -acq.GetPoint(pt1).GetValues()[i,:]

                    out[i,0] = np.rad2deg(geometry.computeAngle(u1,v1))
                    out[i,0] = - out[i,0] if key[0]=="R" else out[i,0]

            if gapDetected:
                LOGGER.logger.warning("Wand Planar Angle - gap detected")

            btkTools.smartAppendPoint(acq,key,out,PointType="Scalar")

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