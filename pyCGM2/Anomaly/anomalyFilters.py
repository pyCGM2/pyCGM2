""" This module contains pyCGM2 anomaly filters to deal with either an anomaly
detector procedure or an anomaly correction procedure

check out the script : `\Tests\\test_anomalies.py` for examples
"""
import btk
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Anomaly.anomalyDetectionProcedures import AnomalyDetectionProcedure
from pyCGM2.Anomaly.anomalyCorrectionProcedures import AnomalyCorrectionProcedure

from typing import List, Tuple, Dict, Optional, Union
class AnomalyDetectionFilter(object):
    """Anomaly detector filter.

    This filter interfaces with an anomaly detection procedure to identify anomalies in biomechanical data.

    Args:
        acq (btk.btkAcquisition): A BTK acquisition instance containing biomechanical data.
        filename (str): The name of the file associated with the BTK acquisition data.
        procedure (AnomalyDetectionProcedure): An instance of a subclass of AnomalyDetectionProcedure used to detect anomalies.

    Kwargs:
        frameRange (List[int]): A list specifying the range of frames to analyze. 
            The list should contain two elements: the start frame and the end frame.

    Attributes:
        m_procedure (AnomalyDetectionProcedure): The anomaly detection procedure instance.
        m_acq (btk.btkAcquisition): The BTK acquisition instance.
        m_filename (str): The file name associated with the acquisition data.
        options (Dict): Additional keyword arguments passed to the procedure.
    """

    def __init__(self, acq:btk.btkAcquisition, filename:str, procedure:AnomalyDetectionProcedure, **kwargs):
        """Initializes the AnomalyDetectionFilter with the given acquisition data, file name, and procedure."""

        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename
        self.options = kwargs

    def run(self):
        """Executes the anomaly detection filter.

        Runs the specified anomaly detection procedure on the acquisition data and returns the detected anomalies.

        Returns:
            dict: The detected anomalies as returned by the anomaly detection procedure.
        """
        self.m_procedure.run(self.m_acq, self.m_filename, self.options)
        anomaly = self.m_procedure.getAnomaly()
        return anomaly


class AnomalyCorrectionFilter(object):
    """Anomaly correction filter.

    This filter interfaces with an anomaly correction procedure to correct identified anomalies in biomechanical data.

    Args:
        acq (btk.btkAcquisition): A BTK acquisition instance containing biomechanical data.
        filename (str): The name of the file associated with the BTK acquisition data.
        procedure (AnomalyCorrectionProcedure): An instance of a subclass of AnomalyCorrectionProcedure used for correcting anomalies.

    Attributes:
        m_procedure (AnomalyCorrectionProcedure): The anomaly correction procedure instance.
        m_acq (btk.btkAcquisition): The BTK acquisition instance.
        m_filename (str): The file name associated with the acquisition data.
    """

    def __init__(self, acq:btk.btkAcquisition, filename:str, procedure:AnomalyCorrectionProcedure):
        """Initializes the AnomalyCorrectionFilter with the given acquisition data, file name, and procedure."""
        self.m_procedure = procedure

        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename

    def run(self):
        """Executes the anomaly correction filter.

        Runs the specified anomaly correction procedure on the acquisition data and returns the corrected data.

        Returns:
            Any: The result of the anomaly correction procedure, typically an updated acquisition instance or similar.
        """
        out = self.m_procedure.run(self.m_acq, self.m_filename)

        return out
