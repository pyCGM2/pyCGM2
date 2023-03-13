# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
import scipy  as sp

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Model.CGM2 import  cgm
from pyCGM2.Tools import btkTools



class GroundReactionIntegrationFilter(object):
    """
    filter for working with time-normalized ground reaction forces

    Args:
        analysisInstance (pyCGM2.Processing.analysis.Analysis): an `analysis` instance.
        procedure (pyCGM2.Processing.GroundReactionForce.GrfIntegrationProcedures.GrfIntegrationProcedure): a procedure instance

    """

    def __init__(self, analysisInstance, procedure):

        self.m_analysis = analysisInstance
        self.m_procedure = procedure



    def run(self):
        """Run the filter
        """
        self.m_procedure.compute(self.m_analysis)
        return self.m_procedure.centerOfmass

       



        
