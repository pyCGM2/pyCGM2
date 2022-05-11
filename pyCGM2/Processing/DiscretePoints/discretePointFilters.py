# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

"""
A *discrete point* is a value at a specific frame of a cycle.

In this module, through the filter `DiscretePointsFilter`, the goal is to get series of discrete values extracted according a
specific strategy (ie a procedure). For instance, the `BenedettiProcedure` extracts dicrete points
recommanded in Benededdi et al (1998):

**References**:

Benedetti, M. G.; Catani, F.; Leardini, A.; Pignotti, E.; Giannini, S. (1998) Data management in gait analysis for clinical applications. In : Clinical biomechanics (Bristol, Avon), vol. 13, n° 3, p. 204–215. DOI: 10.1016/s0268-0033(97)00041-7.


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

from pyCGM2.Processing import exporter
from pyCGM2.Signal.detect_peaks import detect_peaks
from pyCGM2.Math import derivation


# --- FILTER ----


class DiscretePointsFilter(object):
    """Discrete point filter

    Args:
        discretePointProcedure (pyCGM2.Processing.DiscrePoints.discretePointProcedures.DiscretePointProcedure): a procedure
        analysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
        modelInfo (dict): information about the model
        subjInfo (dict): information about the subject
        condExpInfo (dict): information about the experiment

    """

    def __init__(self, discretePointProcedure, analysis,modelInfo=None,subjInfo=None,condExpInfo=None):

        self.m_procedure = discretePointProcedure
        self.m_analysis=analysis
        self.dataframe =  None

        self.m_modelInfo = modelInfo
        self.m_subjInfo = subjInfo
        self.m_condExpInfo = condExpInfo

    def setModelInfo(self, modelInfo):
        """Set model information

        Args:
            modelInfo (dict): model information

        """
        self.m_modelInfo = modelInfo

    def setSubjInfo(self, subjInfo):
        """Set subject information

        Args:
            subjInfo (dict): subject information

        """
        self.m_subjInfo = subjInfo

    def setCondExpInf(self, condExpInfo):
        """Set experimental information

        Args:
            condExpInfo (dict): experimental information

        """
        self.m_condExpInfo = condExpInfo


    def getOutput(self):
        """ get the output dataframe with all discrete values according to the procedure
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
