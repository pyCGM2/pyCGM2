from pyCGM2.Tools import btkTools
from pyCGM2.Signal import anomaly
from pyCGM2.Math import derivation
from pyCGM2.Signal import signal_processing

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import logging


try:
    from pyCGM2 import btk
except:
    logging.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk



class AbstractDetectionProcedure(object):
    def __init__(self):
        pass

    def run(self,acq,filename):
        pass



class MarkerAnomalyDetectionRollingProcedure(AbstractDetectionProcedure):
    def __init__(self,markers,plot=False,**options):

        super(MarkerAnomalyDetectionRollingProcedure, self).__init__()

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


class GaitEventAnomalyProcedure(AbstractDetectionProcedure):
    def __init__(self):

        super(GaitEventAnomalyProcedure, self).__init__()

    def run(self,acq,filename):

        events = acq.GetEvents()
        if events.GetItemNumber() != 0:

            events_L = list()
            events_R = list()
            for ev in btk.Iterate(events):

                if ev.GetContext() == "Left":
                    events_L.append(ev)
                if ev.GetContext() == "Right":
                    events_R.append(ev)


            if events_L!=[] and events_R!=[]:
                labels = [it.GetLabel() for it in btk.Iterate(events) if it.GetLabel() in ["Foot Strike","Foot Off"]]
                frames = [it.GetFrame() for it in btk.Iterate(events) if it.GetLabel() in ["Foot Strike","Foot Off"]]

                init = labels[0]
                for i in range(1,len(labels)):
                    label = labels[i]
                    frame = frames[i]
                    if label == init:
                        logging.error("[pyCGM2] file (%s) - two consecutive (%s) detected at frame (%i)"%(filename,(label),frame))

                    init = label

            if events_L!=[]:
                init = events_L[0].GetLabel()
                if len(events_L)>1:
                    for i in range(1,len(events_L)):
                        label = events_L[i].GetLabel()
                        if label == init:
                            logging.error("[pyCGM2] file (%s) - Wrong Left Event - two consecutive (%s) detected at frane (%i)"%(filename,(label),events_L[i].GetFrame()) )
                        init = label
                else:
                    logging.warning("Only one left events")

            if events_R!=[]:
                init = events_R[0].GetLabel()
                if len(events_R)>1:
                    for i in range(1,len(events_R)):
                        label = events_R[i].GetLabel()
                        if label == init:
                            logging.error("[pyCGM2] file (%s) - Wrong Right Event - two consecutive (%s) detected at frane (%i)"%(filename,(label),events_R[i].GetFrame()) )
                        init = label
                else:
                    logging.warning("Only one right events ")

        else:
            logging.error("[pyCGM2-Checking] No events are in trial (%s)"%(filename))



class ForcePlateAnomalyProcedure(object):
    def __init__(self):
        super(ForcePlateAnomalyProcedure, self).__init__()

    def run(self,acq,filename):

        # --- ground reaction force wrench ---
        pfe = btk.btkForcePlatformsExtractor()
        grwf = btk.btkGroundReactionWrenchFilter()
        pfe.SetInput(acq)
        pfc = pfe.GetOutput()
        grwf.SetInput(pfc)
        grwc = grwf.GetOutput()
        grwc.Update()

        fp_counter = 1
        for fpIt in btk.Iterate(grwc):
            force_Z = fpIt.GetForce().GetValues()[:,2]

            max =  np.max(force_Z)
            indexes = np.where(force_Z == max)

            if indexes[0].shape[0] > 1:
                logging.warning ("[pyCGM2] - check Force plate (%s) of file [%s] - signal Fz seems saturating" %(str(fp_counter),filename))

            fp_counter +=1


class MarkerPresenceDetectionProcedure(object):
    def __init__(self,markers=None):
        super(MarkerPresenceDetectionProcedure, self).__init__()

        self.markers = markers

    def run(self,acq,filename):

        markersIn = list()

        for marker in self.markers:
            try:
                acq.GetPoint(marker)
            except RuntimeError:
                logging.warning("[pyCGM2-Checking]  marker [%s] - not exist in the file "%(marker, filename))
            else:
                markersIn.append(marker)

        return markersIn
