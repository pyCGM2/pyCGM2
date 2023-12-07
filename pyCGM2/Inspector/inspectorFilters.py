"""
the inspector filter calls procedure for inspecting acquisition 
"""

import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Inspector.inspectorProcedures import InspectorProcedure
import btk

from typing import List, Tuple, Dict, Optional,Union
class InspectorFilter(object):
    """
    Filter for inspecting a biomechanical acquisition using a specified inspection procedure.

    Args:
        acq (btk.btkAcquisition): The btk acquisition to be inspected.
        filename (str): Filename of the C3D file associated with the acquisition.
        procedure (InspectorProcedure): The inspection procedure to be used for inspecting the acquisition.

    
    """
    def __init__(self,acq:btk.btkAcquisition,filename:str,procedure:InspectorProcedure,**kwargs):
        """
        Initializes the InspectorFilter with the specified acquisition, filename, and inspection procedure.
        """

        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename
        self.options = kwargs

    def run(self):
        """
        Executes the filter to inspect the acquisition based on the specified procedure.

        Returns:
            Any: The output of the inspection procedure.
        """
        out = self.m_procedure.run(self.m_acq,self.m_filename,self.options)

        return out
