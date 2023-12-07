"""
This module is designed for classifying model outputs according to specific gait patterns. It follows an adapter pattern, 
where the classification filter invokes a procedure and then returns the identified gait pattern for both limbs.

Example usage:

.. code-block:: python

    from pyCGM2.Processing.Classification import classificationFilters, classificationProcedures
    from pyCGM2.Report import normativeDatasets

    nds = normativeDatasets.NormativeData("Schwartz2008","Free")

    procedure = classificationProcedures.PFKEprocedure(nds)
    filt = classificationFilters.ClassificationFilter(analysisInstance, procedure)
    sagClass = filt.run()

"""

import pyCGM2
LOGGER = pyCGM2.LOGGER

from pyCGM2.Processing.Classification.classificationProcedures import ClassificationProcedure
from pyCGM2.Processing.analysis import Analysis
from typing import List, Tuple, Dict, Optional,Union,Any



class ClassificationFilter(object):
    """
    A filter for classifying biomechanical model outputs based on gait analysis.

    This filter uses a specified procedure to classify the gait data from an analysis instance.

    Args:
        analysisInstance (pyCGM2.Processing.analysis.Analysis): An instance of the analysis containing biomechanical data.
        procedure (pyCGM2.Processing.Classification.classificationProcedures.ClassificationProcedure): 
                  The procedure to be used for classification.
        pointSuffix (Optional[str]): Suffix to be added to model outputs, default is None.

    """

    def __init__(self, analysisInstance:Analysis, procedure:ClassificationProcedure,pointSuffix:Optional[str]=None):

        self.m_procedure = procedure
        self.m_analysis = analysisInstance
        self.m_pointSuffix = pointSuffix

    def run(self):
        """
        Execute the classification procedure on the analysis data.

        Returns:
            Dict: Classification results for both limbs.
        """

        classification = self.m_procedure.run(self.m_analysis,self.m_pointSuffix)

        return classification
