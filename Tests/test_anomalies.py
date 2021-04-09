# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_anomalies.py::Test_markerAnomalies::test_anomalies
import logging

import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.Anomaly import AnomalyFilter, AnomalyDetectionProcedure, AnomalyCorrectionProcedure

from pyCGM2.Model.CGM2 import cgm

class Test_markerAnomalies:
    def test_noAnomalies_gaitCGM1(self):

        filename = pyCGM2.TEST_DATA_PATH+"LowLevel/anomalies/noOutliers_gaitCGM1/gait Trial 01.c3d"

        markers = cgm.CGM1.LOWERLIMB_TRACKING_MARKERS

        acq = btkTools.smartReader(filename)

        madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure( markers, plot=False, window=10,threshold = 3)
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,filename[filename.rfind("/")+1:],madp)
        anomaly = adf.run()
        anomalyIndexes = anomaly["Output"]

        macp = AnomalyCorrectionProcedure.MarkerAnomalyCorrectionProcedure(markers,anomalyIndexes,plot=False,distance_threshold=20)
        acf = AnomalyFilter.AnomalyCorrectionFilter(acq,filename[filename.rfind("/")+1:],macp)
        acqo = acf.run()


    def test_anomalies(self):

        filename = pyCGM2.TEST_DATA_PATH+"LowLevel/anomalies/multiShortSwapping/test 06.c3d"
        markers = "Y"

        acq = btkTools.smartReader(filename)

        madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure( markers, plot=False, window=10)
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,filename[filename.rfind("/")+1:],madp)
        anomaly = adf.run()
        anomalyIndexes = anomaly["Output"]

        macp = AnomalyCorrectionProcedure.MarkerAnomalyCorrectionProcedure(markers,anomalyIndexes,plot=True,distance_threshold=20)
        acf = AnomalyFilter.AnomalyCorrectionFilter(acq,filename[filename.rfind("/")+1:],macp)
        acqo = acf.run()


class Test_eventsAnomalies:
    def test_gaitEventsAnomalies(self):

        filename = pyCGM2.TEST_DATA_PATH+"/LowLevel/anomalies/gaitEvents/gait Trial 01-noAnomalies.c3d"
        acq = btkTools.smartReader(filename)
        madp = AnomalyDetectionProcedure.GaitEventAnomalyProcedure()
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,filename,madp)
        adf.run()

        filename = pyCGM2.TEST_DATA_PATH+"/LowLevel/anomalies/gaitEvents/gait Trial 01-noEvents.c3d"
        acq = btkTools.smartReader(filename)
        madp = AnomalyDetectionProcedure.GaitEventAnomalyProcedure()
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,filename[filename.rfind("/")+1:],madp)
        adf.run()

        filename = pyCGM2.TEST_DATA_PATH+"/LowLevel/anomalies/gaitEvents/gait Trial 01-LeftDoubleFS_anomalies.c3d"
        acq = btkTools.smartReader(filename)
        madp = AnomalyDetectionProcedure.GaitEventAnomalyProcedure()
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,filename[filename.rfind("/")+1:],madp)
        adf.run()

        filename = pyCGM2.TEST_DATA_PATH+"/LowLevel/anomalies/gaitEvents/gait Trial 01-LeftDoubleFO_anomalies.c3d"
        acq = btkTools.smartReader(filename)
        madp = AnomalyDetectionProcedure.GaitEventAnomalyProcedure()
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,filename[filename.rfind("/")+1:],madp)
        adf.run()

class Test_ForcePlateAnomalies:
    def test_forceAnomalies(self):

        filename = pyCGM2.TEST_DATA_PATH+"/LowLevel/anomalies/gaitEvents/gait Trial 01-noAnomalies.c3d"
        acq = btkTools.smartReader(filename)
        madp = AnomalyDetectionProcedure.ForcePlateAnomalyProcedure()
        adf = AnomalyFilter.AnomalyDetectionFilter(acq,filename[filename.rfind("/")+1:],madp)
        adf.run()
