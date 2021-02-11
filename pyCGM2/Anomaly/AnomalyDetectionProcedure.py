from pyCGM2.Tools import btkTools
from pyCGM2.Signal import anomaly
from pyCGM2.Math import derivation
from pyCGM2.Signal import signal_processing

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import logging


class MarkerAnomalyDetectionRollingProcedure(object):
    def __init__(self,markers,plot=False,**options):

        if type(markers) == str:
            markers = [markers]

        self.m_markers = markers
        self._plot = plot

        self._aprioriError = 3 if "aprioriError" not in options else options["aprioriError"]
        self._window = 10 if "window" not in options else options["window"]
        self._treshold = 3 if "treshold" not in options else options["treshold"]
        self._method = "median" if "method" not in options else options["method"]



    def run(self,acq,filename):
        out = dict()

        ff = acq.GetFirstFrame()

        for marker in self.m_markers:
            pointValues = acq.GetPoint(marker).GetValues()


            values = pointValues[:,0] #np.linalg.norm(pointValues,axis=1)
            # values = derivation.firstOrderFiniteDifference(pointValues,100)[:,2]

            # values = signal_processing.arrayLowPassFiltering(pointValues, 100)[:,2]q

            indices0 = anomaly.anomaly_rolling(values,
                                        aprioriError= self._aprioriError,
                                        label= marker + " ["+filename+"]",
                                        window = self._window,
                                        threshold =self._treshold,
                                        method = self._method,
                                        plot=self._plot,
                                        referenceValues = None)
            indices_frameMatched0 = [it+ff for it in indices0]


            values = pointValues[:,1] #np.linalg.norm(pointValues,axis=1)
            # values = derivation.firstOrderFiniteDifference(pointValues,100)[:,2]

            # values = signal_processing.arrayLowPassFiltering(pointValues, 100)[:,2]q

            indices1 = anomaly.anomaly_rolling(values,
                                        aprioriError= self._aprioriError,
                                        label= marker + " ["+filename+"]",
                                        window = self._window,
                                        threshold =self._treshold,
                                        method = self._method,
                                        plot=self._plot,
                                        referenceValues = None)
            indices_frameMatched1 = [it+ff for it in indices1]


            values = pointValues[:,2] #np.linalg.norm(pointValues,axis=1)
            # values = derivation.firstOrderFiniteDifference(pointValues,100)[:,2]

            # values = signal_processing.arrayLowPassFiltering(pointValues, 100)[:,2]

            indices2 = anomaly.anomaly_rolling(values,
                                        aprioriError= self._aprioriError,
                                        label= marker + " ["+filename+"]",
                                        window = self._window,
                                        threshold =self._treshold,
                                        method = self._method,
                                        plot=self._plot,
                                        referenceValues = None)
            indices_frameMatched2 = [it+ff for it in indices2]

            indices_frameMatchedAll = []
            if indices_frameMatched0!=[] and indices_frameMatched1 !=[] and indices_frameMatched2 != []:
                allIndexes= indices_frameMatched1 + indices_frameMatched2

                [indices_frameMatchedAll.append(x) for x in allIndexes if x not in indices_frameMatchedAll]


            if indices_frameMatchedAll !=[]:
                logging.warning("[pyCGM2] marker %s [file : %s]- anomalies found at %s"% (marker,filename,indices_frameMatchedAll))
            out[marker] = indices_frameMatchedAll
        return out
