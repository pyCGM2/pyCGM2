""" This module gathers anomaly detectors on markers, events, force plate signals and anthropometric data

check out the script : `\\Tests\\test_anomalies.py` for example

"""

from pyCGM2.Signal import anomaly
from pyCGM2.Utils import utils

import numpy as np

import pyCGM2
LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional, Union

import btk

class AnomalyDetectionProcedure(object):
    """Abstract base class for anomaly detection procedures.

    This class provides a basic structure for implementing different anomaly detection techniques.
    It should be extended to add specific functionalities for various types of anomaly detection.
    """

    def __init__(self):
        """Initializes the AnomalyDetectionProcedure class."""
        self.anomaly = {"Output": None,
                        "ErrorState": False}

    def run(self, acq:btk.btkAcquisition, filename:str, options:Dict):
        """Abstract method to run the anomaly detection procedure.

        Args:
            acq (btk.btkAcquisition): An instance of a btk acquisition object.
            filename (str): The filename of the data to be processed.
            options (Dict): A dictionary of options specific to the detection procedure.
        """
        pass

    def getAnomaly(self):
        """Returns the detected anomaly.

        Returns:
            dict: A dictionary containing the anomaly output and error state.
        """
        return self.anomaly


class MarkerAnomalyDetectionRollingProcedure(AnomalyDetectionProcedure):
    """Marker anomaly detection using rolling statistics.

    This class implements a procedure for detecting anomalies in marker trajectories using rolling statistics.

    Attributes:
        m_markers (List): List of marker labels.
        _plot (bool): Flag indicating whether to enable plotting.
        _aprioriError (float): A priori error on the marker trajectory.
        _window (int): Size of the rolling windows.
        _treshold (int): Detector threshold associated with the standard deviation.
        _method (str): Method to use for rolling statistics, either 'mean' or 'median'.
    """

    def __init__(self, markers:List, plot:bool=False, **kwargs):
        """
        Initialize the MarkerAnomalyDetectionRollingProcedure class with given parameters.

        Args:
            markers (List): List of marker labels.
            plot (bool, optional): Flag indicating whether to enable plotting. Defaults to False.
            **kwargs: Additional keyword arguments including aprioriError, window, treshold, and method.
        """

        super(MarkerAnomalyDetectionRollingProcedure, self).__init__()

        if type(markers) == str:
            markers = [markers]

        self.m_markers = markers
        self._plot = plot

        self._aprioriError = 3 if "aprioriError" not in kwargs else kwargs["aprioriError"]
        self._window = 10 if "window" not in kwargs else kwargs["window"]
        self._treshold = 3 if "treshold" not in kwargs else kwargs["treshold"]
        self._method = "median" if "method" not in kwargs else kwargs["method"]

    def run(self, acq:btk.btkAcquisition, filename:str, options:Dict):
        """Run the marker anomaly detection procedure.

        Args:
            acq (btk.btkAcquisition): A btk acquisition instance.
            filename (str): Filename of the data being processed.
            options (Dict): Additional options passed from the filter.
        """

        errorState = False

        out = {}

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
    """Gait event anomaly detector.

    This class implements a procedure for detecting anomalies in gait events.

    Inherits from AnomalyDetectionProcedure.
    """

    def __init__(self):
        """Initializes the GaitEventAnomalyProcedure class."""
        super(GaitEventAnomalyProcedure, self).__init__()

    def run(self, acq:btk.btkAcquisition, filename:str, options:Dict):
        """Run the gait event anomaly detection procedure.

        This method analyzes gait events and detects any anomalies, such as two consecutive identical events.

        Args:
            acq (btk.btkAcquisition): An instance of a btk acquisition object.
            filename (str): The filename of the data being processed.
            options (Dict): Additional options passed from the filter.
        """

        errorState = False

        events = acq.GetEvents()
        if events.GetItemNumber() != 0:

            events_L = []
            events_R = []
            for ev in btk.Iterate(events):

                if ev.GetContext() == "Left":
                    events_L.append(ev)
                if ev.GetContext() == "Right":
                    events_R.append(ev)

            if events_L != [] and events_R != []:
                labels = [it.GetLabel() for it in btk.Iterate(
                    events) if it.GetLabel() in ["Foot Strike", "Foot Off"] and it.GetContext() in ["Left","Right"]]
                frames = [it.GetFrame() for it in btk.Iterate(
                    events) if it.GetLabel() in ["Foot Strike", "Foot Off"] and it.GetContext() in ["Left","Right"]]
                                
                # sort
                asso = sorted(zip(frames, labels))
                frames_sorted, labels_sorted = zip(*asso)
                frames_sorted = list(frames_sorted)
                labels_sorted = list(labels_sorted)
                
                init = labels_sorted[0]
                for i in range(1, len(labels_sorted)):
                    label = labels_sorted[i]
                    frame = frames_sorted[i]
                    if label == init:
                        LOGGER.logger.warning(
                            "[pyCGM2-Anomaly] file (%s) - two consecutive (%s) detected at frame (%i)" % (filename, (label), frame))
                        errorState = True

                    init = label
            

            if events_L != []:

                if len(events_L) > 1:
                    labels = [it.GetLabel() for it in btk.Iterate(
                        events) if it.GetLabel() in ["Foot Strike", "Foot Off"] and it.GetContext() in ["Left"]]
                    frames = [it.GetFrame() for it in btk.Iterate(
                        events) if it.GetLabel() in ["Foot Strike", "Foot Off"] and it.GetContext() in ["Left"]]
                                    
                    # sort
                    asso = sorted(zip(frames, labels))
                    frames_sorted, labels_sorted = zip(*asso)
                    frames_sorted = list(frames_sorted)
                    labels_sorted = list(labels_sorted)

                    init = labels_sorted[0]
                    for i in range(1, len(labels_sorted)):
                        label = labels_sorted[i]
                        frame = frames_sorted[i]
                        if label == init:
                            LOGGER.logger.warning(
                                "[pyCGM2-Anomaly] file (%s) - Wrong Left Event - two consecutive (%s) detected at frame (%i)" % (filename, (label), frame))
                            errorState = True

                        init = label
                else:
                    LOGGER.logger.warning(
                        "[pyCGM2-Anomaly] Only one left events")



                # init = events_L[0].GetLabel()
                # if len(events_L) > 1:
                #     for i in range(1, len(events_L)):
                #         label = events_L[i].GetLabel()

                #         if label == init:
                #             LOGGER.logger.warning("[pyCGM2-Anomaly] file (%s) - Wrong Left Event - two consecutive (%s) detected at frane (%i)" % (
                #                 filename, (label), events_L[i].GetFrame()))
                #             errorState = True
                #         init = label
                # else:
                #     LOGGER.logger.warning(
                #         "[pyCGM2-Anomaly] Only one left events")

            if events_R != []:
                if len(events_R) > 1:
                    labels = [it.GetLabel() for it in btk.Iterate(
                        events) if it.GetLabel() in ["Foot Strike", "Foot Off"] and it.GetContext() in ["Right"]]
                    frames = [it.GetFrame() for it in btk.Iterate(
                        events) if it.GetLabel() in ["Foot Strike", "Foot Off"] and it.GetContext() in ["Right"]]
                                    
                    # sort
                    asso = sorted(zip(frames, labels))
                    frames_sorted, labels_sorted = zip(*asso)
                    frames_sorted = list(frames_sorted)
                    labels_sorted = list(labels_sorted)

                    init = labels_sorted[0]
                    for i in range(1, len(labels_sorted)):
                        label = labels_sorted[i]
                        frame = frames_sorted[i]
                        if label == init:
                            LOGGER.logger.warning(
                                "[pyCGM2-Anomaly] file (%s) - Wrong Right Event - two consecutive (%s) detected at frame (%i)" % (filename, (label), frame))
                            errorState = True

                        init = label
                else:
                    LOGGER.logger.warning(
                        "[pyCGM2-Anomaly] Only one right events ")

        else:
            LOGGER.logger.warning(
                "[pyCGM2-Anomaly] No gait events are in the trial (%s)" % (filename))
            self.anomaly["Output"] = "No gait events"

        self.anomaly["ErrorState"] = errorState
        return errorState


class ForcePlateAnomalyProcedure(AnomalyDetectionProcedure):
    """Force plate anomaly detector.

    This class is designed to detect anomalies in force plate data, such as saturated signals.

    Inherits from AnomalyDetectionProcedure.
    """

    def __init__(self):
        """Initializes the ForcePlateAnomalyProcedure class."""
        super(ForcePlateAnomalyProcedure, self).__init__()

    def run(self, acq:btk.btkAcquisition, filename:str, options:Dict):

        """Run the force plate anomaly detection procedure.

        This method checks force plate signals for anomalies like saturation. The analysis can be customized using the `options` dictionary.

        Args:
            acq (btk.btkAcquisition): A btk acquisition instance.
            filename (str): Filename of the data being processed.
            options (Dict): Additional options passed from the filter. It can include:
                - 'frameRange' (List[int, int]): A list of two integers specifying the start and end frames for the analysis.
                If not provided, the analysis uses the full range of frames available in the acquisition data.

        Note: Other options may be included in the dictionary, but they are not currently used in this implementation.
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
    """Anthropometric data anomaly detector.

    This class implements a procedure for detecting anomalies in anthropometric data, such as implausible body measurements.

    Inherits from AnomalyDetectionProcedure.

    Attributes:
        mp (Dict): A dictionary containing anthropometric measurements.
    """

    def __init__(self, mp):
        """
        Initializes the AnthropoDataAnomalyProcedure class with given anthropometric measurements.

        Args:
            mp (Dict): A dictionary containing anthropometric measurements.
        """
        super(AnthropoDataAnomalyProcedure, self).__init__()

        self.mp = mp

    def run(self, acq:btk.btkAcquisition, filename:str, options:Dict):
        """Run the anthropometric data anomaly detection procedure.

        This method checks for anomalies in anthropometric data, such as extremely low body mass or disproportionate limb measurements. The `options` parameter is included for future extensibility but is not currently used in the implementation.

        Args:
            acq (btk.btkAcquisition): A btk acquisition instance.
            filename (str): Filename of the data being processed.
            options (Dict): Additional options passed from the filter. Currently, this parameter is not used in the implementation.

        Note: Future versions of this method may use the `options` dictionary to provide additional customization and functionality.
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
