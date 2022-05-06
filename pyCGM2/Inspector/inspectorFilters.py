# -*- coding: utf-8 -*-
#APIDOC: /Low level/Inspector
"""
Module contain the filter for running inspector procedure.
"""
import pyCGM2; LOGGER = pyCGM2.LOGGER

class InspectorFilter(object):
    """
    pyCGM2 filter
    """
    def __init__(self,acq,filename,procedure,**kwargs):
        """Constructor

        Args:
            acq (btk.Acquisition): a btk acquisition
            filename (str): c3d filename
            procedure (pyCGM2.Inspector.InspectorProcedure): an inspector procedure instance


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
