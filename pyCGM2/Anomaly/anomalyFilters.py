""" This module contains pyCGM2 anomaly filters to deal with either an anomaly
detector procedure or an anomaly correction procedure

check out the script : `\Tests\\test_anomalies.py` for examples
"""
import btk
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Anomaly.anomalyDetectionProcedures import AnomalyDetectionProcedure
from pyCGM2.Anomaly.anomalyCorrectionProcedures import AnomalyCorrectionProcedure

class AnomalyDetectionFilter(object):
    """ Anomaly detector filter

    Args:
        acq (btk.Acquisition): a btk acquisition instance
        filename (str): filename
        procedure(AnomalyDetectionProcedure): an AnomalyDetectionProcedure procedure instance

    Kwargs:
        frameRange (list): frame boundaries
    """

    def __init__(self, acq:btk.btkAcquisition, filename:str, procedure:AnomalyDetectionProcedure, **kwargs):
        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename
        self.options = kwargs

    def run(self):
        """ run the filter"""
        self.m_procedure.run(self.m_acq, self.m_filename, self.options)
        anomaly = self.m_procedure.getAnomaly()
        return anomaly


class AnomalyCorrectionFilter(object):
    """ anomaly corrector filter

    Args:
        acq (btk.Acquisition): a btk acquisition instance
        filename (str): filename
        procedure(AnomalyCorrectionProcedure): a AnomalyCorrectionProcedure procedure instance
    """

    def __init__(self, acq:btk.btkAcquisition, filename:str, procedure:AnomalyCorrectionProcedure):
        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename

    def run(self):
        """ run the filter"""
        out = self.m_procedure.run(self.m_acq, self.m_filename)

        return out
