"""
This module focuses on extracting discrete points from biomechanical data cycles. 
Discrete points are specific values at particular frames within a cycle. 
Using the DiscretePointsFilter, this module enables the extraction of discrete values 
according to a specified strategy defined by a procedure. An example procedure used 
is the BenedettiProcedure, which follows the recommendations from Benedetti et al (1998) 
for gait analysis in clinical applications.

References:
Benedetti, M. G.; Catani, F.; Leardini, A.; Pignotti, E.; Giannini, S. (1998) 
Data management in gait analysis for clinical applications. 
Clinical Biomechanics (Bristol, Avon), vol. 13, n° 3, p. 204–215. DOI: 10.1016/s0268-0033(97)00041-7.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

from pyCGM2.Processing import exporter
from pyCGM2.Signal.detect_peaks import detect_peaks
from pyCGM2.Math import derivation

from pyCGM2.Processing.DiscretePoints.discretePointProcedures import DiscretePointProcedure
from pyCGM2.Processing.analysis import Analysis
from typing import List, Tuple, Dict, Optional,Union,Any

# --- FILTER ----


class DiscretePointsFilter(object):
    """
    Filter to extract discrete points from biomechanical data cycles.

    Args:
        discretePointProcedure (DiscretePointProcedure): Procedure to extract discrete points.
        analysis (Analysis): Analysis instance containing biomechanical data.
        modelInfo (Optional[Dict]): Information about the biomechanical model.
        subjInfo (Optional[Dict]): Information about the subject.
        condExpInfo (Optional[Dict]): Information about the experimental conditions.
    """

    def __init__(self, discretePointProcedure:DiscretePointProcedure, 
                 analysis:Analysis,
                 modelInfo:Optional[Dict]=None,
                 subjInfo:Optional[Dict]=None,
                 condExpInfo:Optional[Dict]=None):

        self.m_procedure = discretePointProcedure
        self.m_analysis=analysis
        self.dataframe =  None

        self.m_modelInfo = modelInfo
        self.m_subjInfo = subjInfo
        self.m_condExpInfo = condExpInfo

    def setModelInfo(self, modelInfo:Dict):
        """
        Set model information.

        Args:
            modelInfo (Dict): Model information.
        """
        self.m_modelInfo = modelInfo

    def setSubjInfo(self, subjInfo:Dict):
        """
        Set subject information.

        Args:
            subjInfo (Dict): Subject information.
        """
        self.m_subjInfo = subjInfo

    def setCondExpInf(self, condExpInfo:Dict):
        """
        Set experimental condition information.

        Args:
            condExpInfo (Dict): Experimental condition information.
        """
        self.m_condExpInfo = condExpInfo


    def getOutput(self):
        """
        Get the output DataFrame with all discrete values according to the procedure.

        Returns:
            pd.DataFrame: DataFrame containing the discrete values extracted.
        """
        self.dataframe = self.m_procedure.detect(self.m_analysis)

        # add infos
        if self.m_modelInfo is not None:
            for key,value in self.m_modelInfo.items():
                exporter.isColumnNameExist( self.dataframe, key)
                self.dataframe[key] = value

        if self.m_subjInfo is not None:
            for key,value in self.m_subjInfo.items():
                exporter.isColumnNameExist( self.dataframe, key)
                self.dataframe[key] = value

        if self.m_condExpInfo is not None:
            for key,value in self.m_condExpInfo.items():
                exporter.isColumnNameExist( self.dataframe, key)
                self.dataframe[key] = value

        return self.dataframe
