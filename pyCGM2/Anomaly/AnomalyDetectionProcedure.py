from pyCGM2.Tools import btkTools
from pyCGM2.Signal import anomaly

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


        self._window = 10 if "window" not in options else options["window"]
        self._treshold = 3 if "treshold" not in options else options["treshold"]
        self._method = "median" if "method" not in options else options["method"]



    def run(self,acq,filename):
        out = dict()

        ff = acq.GetFirstFrame()

        for marker in self.m_markers:
            pointValues = acq.GetPoint(marker).GetValues()

            values = pointValues[:,2]
            #np.linalg.norm(pointValues,axis=1)


            indices = anomaly.anomaly_rolling(values,
                                        label= marker + " ["+filename+"]",
                                        window = self._window,
                                        threshold =self._treshold,
                                        method = self._method,
                                        plot=self._plot)
            indices_frameMatched = [it+ff for it in indices]

            if indices !=[]:
                logging.warning("[pyCGM2] marker %s [file : %s]- anomalies found at %s"% (marker,filename,indices_frameMatched))
            out[marker] = indices_frameMatched
        return out
