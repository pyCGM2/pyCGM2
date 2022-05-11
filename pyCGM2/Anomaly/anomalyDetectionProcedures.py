# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Anomaly
#APIDOC["Draft"]=False
#--end--

""" This module gathers anomaly detectors on markers, events, force plate signals and anthropometric data

check out the script : `\\Tests\\test_anomalies.py` for example

"""

from pyCGM2.Signal import anomaly
from pyCGM2.Utils import utils

import numpy as np

import pyCGM2
LOGGER = pyCGM2.LOGGER


try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")

class AnomalyDetectionProcedure(object):
    """abstract marker detector procedure """

    def __init__(self):
        self.anomaly = {"Output": None,
                        "ErrorState": False}

    def run(self, acq, filename, options):
        pass

    def getAnomaly(self):
        return self.anomaly


class MarkerAnomalyDetectionRollingProcedure(AnomalyDetectionProcedure):
    """marker anomaly detection from rolling statistics

    Args:
        markers (list): marker labels;
        plot (bool): enable plot

    Kwargs:
        aprioriError (double): a priori error on the marker trajectory
        window (int): size of the rolling windows
        treshold (int) : detector threshold assoiated to the standard deviation
        method (str) : mean or median
    """

    def __init__(self, markers, plot=False, **kwargs):

        super(MarkerAnomalyDetectionRollingProcedure, self).__init__()

        if type(markers) == str:
            markers = [markers]

        self.m_markers = markers
        self._plot = plot

        self._aprioriError = 3 if "aprioriError" not in kwargs else kwargs["aprioriError"]
        self._window = 10 if "window" not in kwargs else kwargs["window"]
        self._treshold = 3 if "treshold" not in kwargs else kwargs["treshold"]
        self._method = "median" if "method" not in kwargs else kwargs["method"]

    def run(self, acq, filename, options):
        """ run the procedure

        Args:
            acq (btk.Acquisition): a btk acquisition instantce
            filename (str): filename
            options (dict) : passed options from the filter

        """

        errorState = False

        out = dict()

        ff = acq.GetFirstFrame()
        lf = acq.GetLastFrame()

        if "frameRange" in options.keys():
            frameInit = options["frameRange"][0]-ff
            frameEnd = options["frameRange"][1]-ff+1
        else:
            frameInit = ff-ff
            frameEnd = lf-ff+1

        for marker in self.m_markers:
            pointValues = acq.GetPoint(marker).GetValues()[
                                       frameInit:frameEnd, :]

            values = pointValues[:, 0]  # np.linalg.norm(pointValues,axis=1)
            # values = derivation.firstOrderFiniteDifference(pointValues,100)[:,2]

            # values = signal_processing.arrayLowPassFiltering(pointValues, 100)[:,2]q

            indices0 = anomaly.anomaly_rolling(values,
                                               aprioriError=self._aprioriError,
                                               label=marker
                                               + " ["+filename+"]",
                                               window=self._window,
                                               threshold=self._treshold,
                                               method=self._method,
                                               plot=self._plot,
                                               referenceValues=None)
            indices_frameMatched0 = [it+ff for it in indices0]

            values = pointValues[:, 1]  # np.linalg.norm(pointValues,axis=1)
            # values = derivation.firstOrderFiniteDifference(pointValues,100)[:,2]

            # values = signal_processing.arrayLowPassFiltering(pointValues, 100)[:,2]q

            indices1 = anomaly.anomaly_rolling(values,
                                               aprioriError=self._aprioriError,
                                               label=marker
                                               + " ["+filename+"]",
                                               window=self._window,
                                               threshold=self._treshold,
                                               method=self._method,
                                               plot=self._plot,
                                               referenceValues=None)
            indices_frameMatched1 = [it+ff for it in indices1]

            values = pointValues[:, 2]  # np.linalg.norm(pointValues,axis=1)
            # values = derivation.firstOrderFiniteDifference(pointValues,100)[:,2]

            # values = signal_processing.arrayLowPassFiltering(pointValues, 100)[:,2]

            indices2 = anomaly.anomaly_rolling(values,
                                               aprioriError=self._aprioriError,
                                               label=marker
                                               + " ["+filename+"]",
                                               window=self._window,
                                               threshold=self._treshold,
                                               method=self._method,
                                               plot=self._plot,
                                               referenceValues=None)
            indices_frameMatched2 = [it+ff for it in indices2]

            indices_frameMatchedAll = []
            if indices_frameMatched0 != [] and indices_frameMatched1 != [] and indices_frameMatched2 != []:
                allIndexes = indices_frameMatched1 + indices_frameMatched2

                [indices_frameMatchedAll.append(
                    x) for x in allIndexes if x not in indices_frameMatchedAll]

            if indices_frameMatchedAll != []:
                LOGGER.logger.warning("[pyCGM2-Anomaly]  marker %s [file : %s]- anomalies found at %s" % (
                    marker, filename, indices_frameMatchedAll))
                errorState = True
            out[marker] = indices_frameMatchedAll

        self.anomaly["Output"] = out
        self.anomaly["ErrorState"] = errorState


class GaitEventAnomalyProcedure(AnomalyDetectionProcedure):
    """gait event anomaly detector
    """

    def __init__(self):

        super(GaitEventAnomalyProcedure, self).__init__()

    def run(self, acq, filename, options):
        """ run the procedure

        Args:
            acq (btk.Acquisition): a btk acquisition instantce
            filename (str): filename
            options (dict) : passed options ( Not used so far)

        """

        errorState = False

        events = acq.GetEvents()
        if events.GetItemNumber() != 0:

            events_L = list()
            events_R = list()
            for ev in btk.Iterate(events):

                if ev.GetContext() == "Left":
                    events_L.append(ev)
                if ev.GetContext() == "Right":
                    events_R.append(ev)

            if events_L != [] and events_R != []:
                labels = [it.GetLabel() for it in btk.Iterate(
                    events) if it.GetLabel() in ["Foot Strike", "Foot Off"]]
                frames = [it.GetFrame() for it in btk.Iterate(
                    events) if it.GetLabel() in ["Foot Strike", "Foot Off"]]

                init = labels[0]
                for i in range(1, len(labels)):
                    label = labels[i]
                    frame = frames[i]
                    if label == init:
                        LOGGER.logger.warning(
                            "[pyCGM2-Anomaly] file (%s) - two consecutive (%s) detected at frame (%i)" % (filename, (label), frame))
                        errorState = True

                    init = label

            if events_L != []:
                init = events_L[0].GetLabel()
                if len(events_L) > 1:
                    for i in range(1, len(events_L)):
                        label = events_L[i].GetLabel()
                        if label == init:
                            LOGGER.logger.warning("[pyCGM2-Anomaly] file (%s) - Wrong Left Event - two consecutive (%s) detected at frane (%i)" % (
                                filename, (label), events_L[i].GetFrame()))
                            errorState = True
                        init = label
                else:
                    LOGGER.logger.warning(
                        "[pyCGM2-Anomaly] Only one left events")

            if events_R != []:
                init = events_R[0].GetLabel()
                if len(events_R) > 1:
                    for i in range(1, len(events_R)):
                        label = events_R[i].GetLabel()
                        if label == init:
                            LOGGER.logger.warning("[pyCGM2-Anomaly] file (%s) - Wrong Right Event - two consecutive (%s) detected at frane (%i)" % (
                                filename, (label), events_R[i].GetFrame()))
                            errorState = True
                        init = label
                else:
                    LOGGER.logger.warning(
                        "[pyCGM2-Anomaly] Only one right events ")

        else:
            LOGGER.logger.info(
                "[pyCGM2-Anomaly] No events are in trial (%s)" % (filename))
            self.anomaly["Output"] = "No events"

        self.anomaly["ErrorState"] = errorState
        return errorState


class ForcePlateAnomalyProcedure(AnomalyDetectionProcedure):
    """force plate anomaly detector
    """

    def __init__(self):

        super(ForcePlateAnomalyProcedure, self).__init__()

    def run(self, acq, filename, options):
        """ run the procedure

        Args:
            acq (btk.Acquisition): a btk acquisition instantce
            filename (str): filename
            options (dict) : passed options

        Note
            ``frameRange`` ([int, int]) is one key of the ``options`` argument

        """

        ff = acq.GetFirstFrame()
        lf = acq.GetLastFrame()

        if "frameRange" in options.keys():
            frameInit = options["frameRange"][0]-ff
            frameEnd = options["frameRange"][1]-ff+1
        else:
            frameInit = ff-ff
            frameEnd = lf-ff+1

        errorState = False
        indexes = []

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
            force_Z = fpIt.GetForce().GetValues()[frameInit:frameEnd, 2]

            max = np.max(force_Z)
            if max != 0:
                indexes = np.where(force_Z == max)

                if indexes[0].shape[0] > 1:
                    LOGGER.logger.warning(
                        "[pyCGM2-Anomaly] - check Force plate (%s) of file [%s] - signal Fz seems saturated" % (str(fp_counter), filename))
                    errorState = True

            fp_counter += 1

        self.anomaly["Output"] = indexes
        self.anomaly["ErrorState"] = errorState


class AnthropoDataAnomalyProcedure(AnomalyDetectionProcedure):
    """atnthropometric data anomaly detector
    """

    def __init__(self, mp):

        super(AnthropoDataAnomalyProcedure, self).__init__()

        self.mp = mp

    def run(self, acq, filename, options):
        """ run the procedure

        Args:
            acq (btk.Acquisition): a btk acquisition instantce
            filename (str): filename
            options (dict) : passed options ( Not used so far)


        """

        errorState = False

        if self.mp["Bodymass"] < 15:
            LOGGER.logger.warning("[pyCGM2-Anomaly] Bodymass < 15 kg ")
            errorState = True
        if self.mp["RightLegLength"] < 400:
            LOGGER.logger.warning("[pyCGM2-Anomaly] Right Leg Lenth < 400 mm")
            errorState = True
        if self.mp["LeftLegLength"] < 400:
            LOGGER.logger.warning("[pyCGM2-Anomaly] Left Leg Lenth < 400 mm")
            errorState = True
        if self.mp["RightKneeWidth"] < self.mp["RightAnkleWidth"]:
            LOGGER.logger.error(
                "[pyCGM2-Anomaly] Right ankle width > knee width ")
            errorState = True
        if self.mp["LeftKneeWidth"] < self.mp["LeftAnkleWidth"]:
            LOGGER.logger.error(
                "[pyCGM2-Anomaly] Right ankle width > knee width ")
            errorState = True
        if self.mp["RightKneeWidth"] > self.mp["RightLegLength"]:
            LOGGER.logger.error(
                "[pyCGM2-Anomaly]  Right knee width > leg length ")
            errorState = True
        if self.mp["LeftKneeWidth"] > self.mp["LeftLegLength"]:
            LOGGER.logger.error(
                " [pyCGM2-Anomaly] Left knee width > leg length ")
            errorState = True

        if not utils.isInRange(self.mp["RightKneeWidth"],
                               self.mp["LeftKneeWidth"]-0.3
                               * self.mp["LeftKneeWidth"],
                               self.mp["LeftKneeWidth"]+0.3*self.mp["LeftKneeWidth"]):
            LOGGER.logger.warning(
                "[pyCGM2-Anomaly] Knee widths differed by more than 30%")
            errorState = True

        if not utils.isInRange(self.mp["RightAnkleWidth"],
                               self.mp["LeftAnkleWidth"]-0.3
                               * self.mp["LeftAnkleWidth"],
                               self.mp["LeftAnkleWidth"]+0.3*self.mp["LeftAnkleWidth"]):
            LOGGER.logger.warning(
                "[pyCGM2-Anomaly] Ankle widths differed by more than 30%")
            errorState = True

        if not utils.isInRange(self.mp["RightLegLength"],
                               self.mp["LeftLegLength"]-0.3
                               * self.mp["LeftLegLength"],
                               self.mp["LeftLegLength"]+0.3*self.mp["LeftLegLength"]):
            LOGGER.logger.warning(
                "[pyCGM2-Anomaly] Leg lengths differed by more than 30%")
            errorState = True

        # self.anomaly["Output"] =
        self.anomaly["ErrorState"] = errorState


# class MarkerPositionQualityProcedure(object):
#     """
#     TODO :
#     - check medial markers if exist
#     """
#
#
#     def __init__(self,acq,markers = None, title = None):
#         self.acq = acq
#         self.markers = markers if markers is not None else btkTools.GetMarkerNames(acq)
#         self.exceptionMode = False
#         self.state = True
#
#         self.title = "Marker position" if title is None else title
#
#     def check(self):
#
#         frameNumber = self.acq.GetPointFrameNumber()
#
#
#         LASI_values = self.acq.GetPoint("LASI").GetValues()
#         RASI_values = self.acq.GetPoint("RASI").GetValues()
#         LPSI_values = self.acq.GetPoint("LPSI").GetValues()
#         RPSI_values = self.acq.GetPoint("RPSI").GetValues()
#         sacrum_values=(self.acq.GetPoint("LPSI").GetValues() + self.acq.GetPoint("RPSI").GetValues()) / 2.0
#         midAsis_values=(self.acq.GetPoint("LASI").GetValues() + self.acq.GetPoint("RASI").GetValues()) / 2.0
#
#
#         projectedLASI = np.array([LASI_values[:,0],LASI_values[:,1],np.zeros((frameNumber))]).T
#         projectedRASI = np.array([RASI_values[:,0],RASI_values[:,1],np.zeros((frameNumber))]).T
#         projectedLPSI = np.array([LPSI_values[:,0],LPSI_values[:,1],np.zeros((frameNumber))]).T
#         projectedRPSI = np.array([RPSI_values[:,0],RPSI_values[:,1],np.zeros((frameNumber))]).T
#
#
#         for i  in range(0,frameNumber):
#             verts = [
#                 projectedLASI[i,0:2], # left, bottom
#                 projectedRASI[i,0:2], # left, top
#                 projectedRPSI[i,0:2], # right, top
#                 projectedLPSI[i,0:2], # right, bottom
#                 projectedLASI[i,0:2], # right, top
#                 ]
#
#             codes = [Path.MOVETO,
#                      Path.LINETO,
#                      Path.LINETO,
#                      Path.LINETO,
#                      Path.CLOSEPOLY,
#                      ]
#
#             path = Path(verts, codes)
#
#             intersection = geometry.LineLineIntersect(projectedLASI[i,:],projectedLPSI[i,:],projectedRASI[i,:],projectedRPSI[i,:])
#
#
#             if path.contains_point(intersection[0]):
#                 LOGGER.logger.error("[pyCGM2-Anomaly] wrong Labelling of pelvic markers at frame [%i]"%(i))
#                 if self.exceptionMode:
#                     raise Exception("[pyCGM2-Anomaly] wrong Labelling of pelvic markers at frame [%i]"%(i))
#
#                 self.state = False
#             else:
#                 # check marker side
#                 pt1=RASI_values[i,:]
#                 pt2=LASI_values[i,:]
#                 pt3=sacrum_values[i,:]
#                 ptOrigin=midAsis_values[i,:]
#
#                 a1=(pt2-pt1)
#                 a1=np.divide(a1,np.linalg.norm(a1))
#                 v=(pt3-pt1)
#                 v=np.divide(v,np.linalg.norm(v))
#                 a2=np.cross(a1,v)
#                 a2=np.divide(a2,np.linalg.norm(a2))
#
#                 x,y,z,R=frame.setFrameData(a1,a2,"YZX")
#
#                 csFrame_L=frame.Frame()
#                 csFrame_L.setRotation(R)
#                 csFrame_L.setTranslation(RASI_values[i,:])
#
#                 csFrame_R=frame.Frame()
#                 csFrame_R.setRotation(R)
#                 csFrame_R.setTranslation(LASI_values[i,:])
#
#
#                 for marker in self.markers:
#                     residual = self.acq.GetPoint(marker).GetResidual(i)
#
#                     if marker[0] == "L":
#                         local = np.dot(csFrame_L.getRotation().T,self.acq.GetPoint(marker).GetValues()[i,:]-csFrame_L.getTranslation())
#                     if marker[0] == "R":
#                         local = np.dot(csFrame_R.getRotation().T,self.acq.GetPoint(marker).GetValues()[i,:]-csFrame_R.getTranslation())
#                     if residual >0.0:
#                         if marker[0] == "L" and local[1]<0:
#                             LOGGER.logger.error("[pyCGM2-Anomaly] check location of the marker [%s] at frame [%i]"%(marker,i))
#                             self.state = False
#                             if self.exceptionMode:
#                                 raise Exception("[pyCGM2-Anomaly] check location of the marker [%s] at frame [%i]"%(marker,i))
#
#                         if marker[0] == "R" and local[1]>0:
#                             LOGGER.logger.error("[pyCGM2-Anomaly] check location of the marker [%s] at frame [%i]"%(marker,i))
#                             self.state = False
#                             if self.exceptionMode:
#                                 raise Exception("[pyCGM2-Anomaly] check location of the marker [%s] at frame [%i]"%(marker,i))
#                                 self.state = False
#
# class AnthropometricDataQualityProcedure(object):
#     def __init__(self,mp,title=None):
#         self.mp = mp
#         self.state = True
#         self.exceptionMode = False
#
#         self.title = "CGM anthropometric parameters" if title is None else title
#
#     def check(self):
#         """
#         TODO :
#         - use relation between variable ( width/height)
#         - use marker measurement
#         """
#
#         if self.mp["RightLegLength"] < 500: LOGGER.logger.warning("[pyCGM2-Anomaly] Right Leg Lenth < 500 mm");self.state = False
#         if self.mp["LeftLegLength"] < 500: LOGGER.logger.warning("[pyCGM2-Anomaly] Left Leg Lenth < 500 mm");self.state = False
#         if self.mp["RightKneeWidth"] < self.mp["RightAnkleWidth"]: LOGGER.logger.error("[pyCGM2-Anomaly] Right ankle width > knee width ");self.state = False
#         if self.mp["LeftKneeWidth"] < self.mp["LeftAnkleWidth"]: LOGGER.logger.error("[pyCGM2-Anomaly] Right ankle width > knee width ");self.state = False
#         if self.mp["RightKneeWidth"] > self.mp["RightLegLength"]: LOGGER.logger.error("[pyCGM2-Anomaly]  Right knee width > leg length ");self.state = False
#         if self.mp["LeftKneeWidth"] > self.mp["LeftLegLength"]: LOGGER.logger.error(" [pyCGM2-Anomaly] Left knee width > leg length ");self.state = False
#
#
#         if not utils.isInRange(self.mp["RightKneeWidth"],
#             self.mp["LeftKneeWidth"]-0.3*self.mp["LeftKneeWidth"],
#             self.mp["LeftKneeWidth"]+0.3*self.mp["LeftKneeWidth"]):
#             LOGGER.logger.warning("[pyCGM2-Anomaly] Knee widths differed by more than 30%")
#             self.state = False
#
#         if not utils.isInRange(self.mp["RightAnkleWidth"],
#             self.mp["LeftAnkleWidth"]-0.3*self.mp["LeftAnkleWidth"],
#             self.mp["LeftAnkleWidth"]+0.3*self.mp["LeftAnkleWidth"]):
#             LOGGER.logger.warning("[pyCGM2-Anomaly] Ankle widths differed by more than 30%")
#             self.state = False
#
#         if not utils.isInRange(self.mp["RightLegLength"],
#             self.mp["LeftLegLength"]-0.3*self.mp["LeftLegLength"],
#             self.mp["LeftLegLength"]+0.3*self.mp["LeftLegLength"]):
#             LOGGER.logger.warning("[pyCGM2-Anomaly] Leg lengths differed by more than 30%")
#             self.state = False
