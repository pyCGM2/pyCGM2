# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Anomaly
#APIDOC["Draft"]=False
#--end--

""" This module contains pyCGM2 anomaly filters to deal with either an anomaly
detector procedure or an anomaly correction procedure

check out the script : `\Tests\\test_anomalies.py` for examples

"""

import pyCGM2
LOGGER = pyCGM2.LOGGER


class AnomalyDetectionFilter(object):
    """ Anomaly detector filter

    Args:
        acq (btk.Acquisition): a btk acquisition instance
        filename (str): filename
        procedure(pyCGM2.Anomaly.anomalyDetectionProcedures.AnomalyDetectionProcedure): a procedure instance

    Kwargs:
        frameRange (list): frame boundaries
    """

    def __init__(self, acq, filename, procedure, **kwargs):
        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename
        self.options = kwargs

    def run(self):
        self.m_procedure.run(self.m_acq, self.m_filename, self.options)
        anomaly = self.m_procedure.getAnomaly()
        return anomaly


class AnomalyCorrectionFilter(object):
    """ anomaly corrector filter

    Args:
        acq (btk.Acquisition): a btk acquisition instance
        filename (str): filename
        procedure(pyCGM2.Anomaly.anomalyDetectionProcedures.AnomalyDetectionProcedure): a  procedure instance
    """

    def __init__(self, acq, filename, procedure):
        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename

    def run(self):
        out = self.m_procedure.run(self.m_acq, self.m_filename)

        return out
