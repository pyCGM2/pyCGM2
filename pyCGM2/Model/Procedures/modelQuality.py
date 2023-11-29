# -*- coding: utf-8 -*-
"""
This module contains procedures for assessing the quality of a biomechanical model.
It includes classes for different types of quality analysis procedures, such as
analyzing wand angles and calculating SCORE residuals for model segments and joints.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict


import pyCGM2
from  pyCGM2.Math import geometry
from pyCGM2.Tools import btkTools
LOGGER = pyCGM2.LOGGER
from pyCGM2.Model.model import Model
from typing import List, Tuple, Dict, Optional,Union,Any

try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")

class QualityProcedure(object):
    """
    Base class for quality assessment procedures of biomechanical models.
    """
    def __init__(self):
        pass

class WandAngleQualityProcedure(QualityProcedure):
    """
    Procedure for assessing the quality of wand angle measurements in a biomechanical model.

    This procedure calculates planar angles using defined points and detects any gaps in the measurements.
    """
    def __init__(self):
        super(WandAngleQualityProcedure, self).__init__()

        self.m_definitions = {"LTHIplanarAngle":['LKNE', 'LHJC', "LTHI"],
                              "RTHIplanarAngle":['RKNE', 'RHJC', "RTHI"],
                              "LTIBplanarAngle":['LANK', 'LKJC', "LTIB"],
                              "RTIBplanarAngle":['RANK', 'RKJC', "RTIB"]}


    def run(self,acq):
        """
        Runs the wand angle quality assessment procedure on the given acquisition.

        Args:
            acq (btk.Acquisition): The acquisition data to analyze.
        """

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
    """
    Procedure for calculating SCORE residual on specific segments of a biomechanical model.

    This class is used to define and handle SCORE residuals for model evaluation.

    Args:
        model (Model): The biomechanical model to be analyzed.
    """

    def __init__(self,model:Model):
        super(GeneralScoreResidualProcedure, self).__init__()

        self.m_scoreDefinition = {}
        self.model = model

    def setDefinition(self, nodeLabel: str, proxSegLabel: str, distSegLabel: str):
        """
        Sets the definition dictionary for calculating SCORE residuals.

        Args:
            nodeLabel (str): The label of the node for which the SCORE residual is calculated.
            proxSegLabel (str): The label of the proximal segment associated with the node.
            distSegLabel (str): The label of the distal segment associated with the node.
        """

        self.m_scoreDefinition[nodeLabel] = {"proximal": proxSegLabel , "distal": distSegLabel }


class ModelScoreResidualProcedure(QualityProcedure):
    """
    Procedure for calculating SCORE residual for all joints of a biomechanical model.

    This class automatically populates its SCORE definition attribute based on the geometric attributes of the model.

    Args:
        model (Model): The biomechanical model to be analyzed.
    """

    def __init__(self, model:Model):
        super(ModelScoreResidualProcedure, self).__init__()

        self.model = model
        self.m_scoreDefinition = {}

        for it in self.model.m_jointCollection:
            self.m_scoreDefinition[it.m_nodeLabel] = {"proximal": it.m_proximalLabel , "distal": it.m_distalLabel }