# coding: utf-8
# pytest -s --disable-pytest-warnings  test_anomalies.py::Test_markerAnomalies::test_noAnomalies

import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.Anomaly import AnomalyFilter, AnomalyDetectionProcedure, AnomalyCorrectionProcedure
import logging


class Test_markerAnomalies:
    def test_noAnomalies(self):

        filename = pyCGM2.TEST_DATA_PATH+"LowLevel/outliers/noOutliers/gait Trial 01.c3d"
        markers = ["LASI",'RASI']

        acq = btkTools.smartReader(filename)

        madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure( markers, plot=False, window=3)
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,filename,madp)
        anomalyIndexes = adf.run()

        macp = AnomalyCorrectionProcedure.MarkerAnomalyCorrectionProcedure(markers,anomalyIndexes,plot=False,distance_threshold=20)
        acf = AnomalyFilter.AnomalyCorrectionFilter(acq,filename,macp)
        acqo = acf.run()


    def test_anomalies(self):

        filename = pyCGM2.TEST_DATA_PATH+"LowLevel/outliers/multiShortSwapping/test 06.c3d"
        markers = "Y"

        acq = btkTools.smartReader(filename)

        madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure( markers, plot=True, window=10)
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,filename,madp)
        anomalyIndexes = adf.run()

        macp = AnomalyCorrectionProcedure.MarkerAnomalyCorrectionProcedure(markers,anomalyIndexes,plot=True,distance_threshold=20)
        acf = AnomalyFilter.AnomalyCorrectionFilter(acq,filename,macp)
        acqo = acf.run()
