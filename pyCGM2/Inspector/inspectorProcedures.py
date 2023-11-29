
"""
Module for Biomechanical Data Inspection Procedures

This module contains classes and procedures designed to inspect and analyze biomechanical data from acquisitions. These procedures are used for checking data integrity, presence of specific markers, and other aspects of biomechanical data without modifying the original data. It serves as a toolkit for validating and understanding the content of biomechanical acquisitions, ensuring data quality and consistency before processing or analysis.
"""
import btk
from pyCGM2.Tools import btkTools
import pyCGM2; LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional, Union


class InspectorProcedure(object):
    """
    Base class for procedures that inspect the content of a biomechanical acquisition.
    An inspector procedure is designed to analyze and report on the data without altering it.
    """
    def __init__(self):
        """
        Initializes the InspectorProcedure.
        """
        pass


class MarkerPresenceDetectionProcedure(InspectorProcedure):
    """
    Procedure to check the presence of specified markers in a biomechanical acquisition.

    Args:
        markers (Optional[List], optional): A list of marker names to check for presence in the acquisition. Defaults to None.
    """
    def __init__(self,markers:Optional[List]=None):
        """
        Initializes the MarkerPresenceDetectionProcedure with a list of markers to be checked.
        """
        super(MarkerPresenceDetectionProcedure, self).__init__()

        self.markers = markers

    def run(self,acq:btk.btkAcquisition,filename:str,options:Dict)-> Dict:
        """
        Executes the procedure to check for the presence of specified markers in the acquisition.

        Args:
            acq (btk.btkAcquisition): The btk acquisition instance to be inspected.
            filename (str): Filename of the C3D file associated with the acquisition.
            options (Dict): Additional options for the procedure.

        Returns:
            Dict: A dictionary with two keys 'In' and 'Out'. 'In' contains a list of markers present in the acquisition, and 'Out' contains a list of markers not found.
        """

        markersIn = []
        markersOut = []

        for marker in self.markers:
            try:
                acq.GetPoint(marker)
            except RuntimeError:
                markersOut.append(marker)
            else:
                if not btkTools.isPhantom(acq,marker):
                    markersOut.append(marker)
                else:
                    markersIn.append(marker)

        if markersIn !=[] and markersOut!=[]:
            for markerOut in markersOut:
                LOGGER.logger.info("[pyCGM2-Inspector]  marker [%s] - not exist in the file [%s]"%(markerOut, filename))

        return {"In":markersIn, "Out":markersOut}
