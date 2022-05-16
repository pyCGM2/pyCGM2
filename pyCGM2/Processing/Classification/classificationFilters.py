# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

"""
This module aims to classify the model ouputs according a gait pattern.

The implementation is based on a *adapter pattern*. The classication filter calls a procedure, then return the gait pattern for both limbs

example:

.. code-block:: python

    nds = normativeDatasets.NormativeData("Schwartz2008","Free")

    procedure = classificationProcedures.PFKEprocedure(nds)
    filt = classificationFilters.ClassificationFilter(analysisInstance, procedure)
    sagClass = filt.run()

"""

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Report import plot



class ClassificationFilter(object):
    """ Classification Filter

    Args:
        analysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
        procedure (pyCGM2.Processing.Classification.classificationProcedures.ClassificationProcedure): a procedure instance
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
