# -*- coding: utf-8 -*-
#APIDOC: /Low level/Anomaly

""" This module contains pyCGM2 anomaly fiters to deal with either the anomaly detector procedure or
the anomaly correction procedure

check out the script : *\Tests\test_anomalies.py* for examples

"""

import pyCGM2
LOGGER = pyCGM2.LOGGER


class AnomalyDetectionFilter(object):
    """ anomaly detector filter
    """

    def __init__(self, acq, filename, procedure, **kwargs):
        """ constructor

        Args:
            acq (btk.Acquisition): a btk acquisition instantce
            filename (str): filename
            procedure(pyCGM2.Anomaly.AnomalyDetectionProcedure): anomaly detector procedure instance

        Low-level Keyword Args :
            frameRange ([int, int]) frame boundaries


        """
        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename
        self.options = kwargs

    def run(self):
        """ run the filter
        """
        self.m_procedure.run(self.m_acq, self.m_filename, self.options)
        anomaly = self.m_procedure.getAnomaly()
        return anomaly


class AnomalyCorrectionFilter(object):
    """ anomaly corrector filter"""

    def __init__(self, acq, filename, procedure):
        """ constructor

        Args:
            acq (btk.Acquisition): a btk acquisition instantce
            filename (str): filename
            procedure(pyCGM2.Anomaly.AnomalyDetectionProcedure): anomaly correction procedure instance

        """
        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename

    def run(self):
        """ run the fiter
        """
        out = self.m_procedure.run(self.m_acq, self.m_filename)

        return out
