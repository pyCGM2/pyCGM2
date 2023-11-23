"""
The module contains filter and procedure for filling gap

check out the script : *\Tests\test_gap.py* for examples
"""

import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Gap.gapFillingProcedures import GapFillingProcedure
import numpy as np

import btk
from typing import List, Tuple, Dict, Optional,Union

#  --- FILTER -----
class GapFillingFilter(object):
    """
    Gap filter

    Args:
        procedure (pyCGM2.Gap.gapFillingProcedures.GapProcedure): a gap filling procedure instance
        acq (Btk.Acquisition): a btk acquisition instance
    """
    def __init__(self,procedure:GapFillingProcedure,acq:btk.btkAcquisition):

        self.m_aqui = acq
        self.m_procedure = procedure

        self.filledMarkers  = None
        self.filledAcq  = None

    def getFilledAcq(self):
        return self.filledAcq

    def getFilledMarkers(self):
        return self.filledMarkers


    def fill(self,markers:Optional[List[str]]=None):
        """
        fill gap according the specified procedure
        """
        if markers is None:
            filledAcq,filledMarkers = self.m_procedure.fill(self.m_aqui)
        else:
            filledAcq,filledMarkers = self.m_procedure.fill(self.m_aqui,markers=markers)


        self.filledMarkers  = filledMarkers
        self.filledAcq  = filledAcq
