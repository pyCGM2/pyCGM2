"""
The module contains filters and procedures for filling gaps in kinematic data.

Check out the script '*\Tests\test_gap.py*' for examples of usage.
"""

import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Gap.gapFillingProcedures import GapFillingProcedure
import numpy as np

import btk
from typing import List, Tuple, Dict, Optional,Union

#  --- FILTER -----
class GapFillingFilter(object):
    """
    A filter for filling gaps in kinematic marker data.

    This class uses specified gap filling procedures to handle missing or incomplete data in kinematic 
    marker datasets. It supports customized gap filling strategies through different procedure implementations.

    Attributes:
        filledMarkers (List[str]): List of markers that were filled during the gap filling process.
        filledAcq (btk.btkAcquisition): The acquisition instance after the gap filling process.

    Args:
        procedure (GapFillingProcedure): A gap filling procedure instance.
        acq (btk.btkAcquisition): A BTK acquisition instance with kinematic data.

    """
    def __init__(self,procedure:GapFillingProcedure,acq:btk.btkAcquisition):

        self.m_aqui = acq
        self.m_procedure = procedure

        self.filledMarkers  = None
        self.filledAcq  = None

    def getFilledAcq(self):
        """
        Retrieves the acquisition instance after the gap filling process.

        This method returns the BTK acquisition instance that has been processed by the gap filling procedure. 
        It contains the kinematic data with gaps filled based on the specified procedure.

        Returns:
            btk.btkAcquisition: The acquisition instance with filled gaps in the kinematic marker data.
        """
        return self.filledAcq

    def getFilledMarkers(self):
        """
        Retrieves the list of markers that were filled during the gap filling process.

        This method returns a list of marker names for which gaps were filled. It provides insight into which 
        markers in the dataset were incomplete and required gap filling.

        Returns:
            List[str]: A list of marker names that had gaps filled during the process.
        """
        return self.filledMarkers


    def fill(self,markers:Optional[List[str]]=None):

        """
        Fills gaps in marker data according to the specified procedure.

        If markers are not specified, gaps in all markers are filled. The method updates the `filledMarkers` and `filledAcq` attributes 
        with the results of the gap filling process.

        Args:
            markers (Optional[List[str]]): A list of specific marker names to fill gaps in. If None, all markers are processed. Defaults to None.
        """
        if markers is None:
            filledAcq,filledMarkers = self.m_procedure.fill(self.m_aqui)
        else:
            filledAcq,filledMarkers = self.m_procedure.fill(self.m_aqui,markers=markers)


        self.filledMarkers  = filledMarkers
        self.filledAcq  = filledAcq
