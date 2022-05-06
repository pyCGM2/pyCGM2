# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

"""
This module aims to classify gait
"""

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Report import plot



class ClassificationFilter(object):
    """ Classification Filter

    Args:
        analysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
        procedure (pyCGM2.Processing.classification.(procedure)): a procedure instance
        pointSuffix (str): suffix added to model outputs.

    """

    def __init__(self, analysisInstance, procedure,pointSuffix=None):

        self.m_procedure = procedure
        self.m_analysis = analysisInstance
        self.m_pointSuffix = pointSuffix

    def run(self):
        """ Run the filter"""

        classification = self.m_procedure.run(self.m_analysis,self.m_pointSuffix)

        return classification
