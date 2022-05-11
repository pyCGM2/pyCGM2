# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

"""
This module aims to detect a representative cycle

The  filter `RepresentativeCycleFilter` calls a specific procedure, and return
indexes of the representive cycle for the Left and right event contexts

"""
import numpy as np
import pandas as pd

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Model.CGM2 import  cgm
from pyCGM2.Tools import btkTools



class RepresentativeCycleFilter(object):
    """Representative cycle filter

    Args:
        analysisInstance (pyCGM2.Processing.analysis.Analysis): an `analysis` instance.
        representativeProcedure (pyCGM2.Processing.Representative.representativeProcedures.RepresentativeProcedure): a procedure instance


    """

    def __init__(self, analysisInstance, representativeProcedure):

        self.m_procedure = representativeProcedure
        self.m_analysis = analysisInstance



    def run(self):
        """Run the filter
        """

        representativeCycleIndex = self.m_procedure._run(self.m_analysis)

        return representativeCycleIndex
