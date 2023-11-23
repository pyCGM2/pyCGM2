"""
the inspector filter calls procedure for inspecting acquisition 
"""

import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Inspector.inspectorProcedures import InspectorProcedure
import btk

from typing import List, Tuple, Dict, Optional,Union
class InspectorFilter(object):

    def __init__(self,acq:btk.btkAcquisition,filename:str,procedure:InspectorProcedure,**kwargs):
        """Constructor

        Args:
            acq (btk.Acquisition): a btk acquisition
            filename (str): c3d filename
            procedure (InspectorProcedure): an inspector procedure instance


        Low-level Keyword Args:
            **kwargs (): passed arguments to the procedure


        """

        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename
        self.options = kwargs

    def run(self):
        out = self.m_procedure.run(self.m_acq,self.m_filename,self.options)

        return out
