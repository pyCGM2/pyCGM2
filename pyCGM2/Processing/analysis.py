"""
The main goal of this module is to construct an `Analysis` instance. It's a matlab-like structure with
spatio temporal (stp), kinematics, kinetics and emg parameters as attributes.

The implementation is based on a *Builder pattern* .
The `AnalysisFilter` calls a `procedure` and return the final `Analysis` instance

"""


import pyCGM2.Processing.cycle as CGM2cycle

from  pyCGM2.Processing.cycle import Cycles

import pyCGM2
LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional,Union,Any

class AnalysisStructure:
    """
    A structure for organizing and storing analysis data.

    Attributes:
        data (Dict): A dictionary to store main analysis results.
        pst (Dict): A dictionary to store associated spatiotemporal parameters.
        optionalData (Dict): A dictionary to store optional or additional analysis data.
    """
    data = {}
    pst = {}
    optionalData = {}


# ---- PATTERN BUILDER ------

# --- OBJECT TO BUILD-----
class Analysis():
    """
    A class-container for storing analysis data including descriptive statistics of various biomechanical parameters.

    Attributes:
        stpStats (dict): Descriptive statistics of spatiotemporal parameters.
        kinematicStats (AnalysisStructure): Descriptive statistics of kinematics data.
        kineticStats (AnalysisStructure): Descriptive statistics of kinetics data.
        emgStats (AnalysisStructure): Descriptive statistics of EMG data.
        muscleGeometryStats (AnalysisStructure): Descriptive statistics of muscle geometry data.
        muscleDynamicStats (AnalysisStructure): Descriptive statistics of muscle dynamics data.
        gps (Optional[dict]): Gait Profile Score (GPS) data.
        gvs (Optional[dict]): Gait Variable Score (GVS) data.
        coactivations (dict): Coactivation data for muscle pairs.
        subjectInfo (Optional[dict]): Information about the subject.
        experimentalInfo (Optional[dict]): Information about the experiment.
        modelInfo (Optional[dict]): Information about the biomechanical model used.
        emgInfo (Optional[dict]): Additional information about EMG data.
        kinematicInfo (Optional[dict]): Additional information about kinematic data.
        kineticInfo (Optional[dict]): Additional information about kinetic data.
        stpInfo (Optional[dict]): Additional information about spatiotemporal parameters.
        scoreInfo (Optional[dict]): Additional information about scores.
        muscleGeometryInfo (dict): Additional information about muscle geometry.
        muscleDynamicInfo (dict): Additional information about muscle dynamics.
    
    
    Examples:
        If you want to return the mean, sd, and median of the left hip angles, time normalized from left gait events

        >>> analysis.kinematicStats.data["LHipAngles", "Left"]["mean"] #return array(101,3)
        >>> analysis.kinematicStats.data["LHipAngles", "Left"]["sd"] #return array(101,3)
        >>> analysis.kinematicStats.data["LHipAngles", "Left"]["median"] #return array(101,3)

        You can also get all-cycle values from the below code.

        >>> analysis.kinematicStats.data["LHipAngles", "Left"]["values"] #return list (length=number of cycle) of array(101,3)

    """

    def __init__(self):

        self.stpStats = {}
        self.kinematicStats = AnalysisStructure()
        self.kineticStats = AnalysisStructure()
        self.emgStats = AnalysisStructure()
        self.muscleGeometryStats = AnalysisStructure()
        self.muscleDynamicStats = AnalysisStructure()

        self.gps = None
        self.gvs = None
        self.coactivations = {}
        self.subjectInfo = None
        self.experimentalInfo = None
        self.modelInfo = None

        self.emgInfo = None
        self.kinematicInfo = None
        self.kineticInfo = None
        self.stpInfo = None
        self.scoreInfo = None
        self.muscleGeometryInfo = {}
        self.muscleDynamicsInfo = {}

    def setStp(self, inDict:Dict):
        """
        Set spatiotemporal parameters.

        Args:
            inDict (Dict): Dictionary of spatiotemporal parameters.
        """
        self.stpStats = inDict

    def setKinematic(self, data:Dict, pst:Dict={}):
        """
        Set kinematic data.

        Args:
            data (Dict): Kinematic data.
            pst (Dict): Spatiotemporal parameters associated with kinematic data.
        """
        self.kinematicStats.data = data
        self.kinematicStats.pst = pst

    def setKinetic(self, data:Dict, pst:Dict={}, optionalData:Dict={}):
        """
        Set kinetic data.

        Args:
            data (Dict): Kinetic data.
            pst (Dict): Spatiotemporal parameters associated with kinetic data.
            optionalData (Dict): Optional additional kinetic data.
        """
        self.kineticStats.data = data
        self.kineticStats.pst = pst
        self.kineticStats.optionalData = optionalData

    def setEmg(self, data: Dict, pst: Dict = {}):
        """
        Set EMG data.

        Args:
            data (Dict): EMG data.
            pst (Dict): Spatiotemporal parameters associated with EMG data.
        """
        self.emgStats.data = data
        self.emgStats.pst = pst

    def setMuscleGeometry(self, data: Dict, pst: Dict = {}):
        """
        Set muscle geometry data.

        Args:
            data (Dict): Muscle geometry data.
            pst (Dict): Spatiotemporal parameters associated with muscle geometry data.
        """
        self.muscleGeometryStats.data = data
        self.muscleGeometryStats.pst = pst


    def setGps(self, GpsStatsOverall: Dict, GpsStatsContext: Dict):
        """
        Set Gait Profile Score (GPS) data.

        Args:
            GpsStatsOverall (Dict): Overall GPS data.
            GpsStatsContext (Dict): GPS data by context (e.g., left, right).
        """
        self.gps = {}
        self.gps["Overall"] = GpsStatsOverall
        self.gps["Context"] = GpsStatsContext

    def setGvs(self, gvsStats: Dict):
        """
        Set Gait Variable Score (GVS) data.

        Args:
            gvsStats (Dict): GVS data.
        """
        self.gvs = gvsStats

    def setSubjectInfo(self, subjectDict: Dict):
        """
        Set subject information.

        Args:
            subjectDict (Dict): Dictionary containing subject information.
        """
        self.subjectInfo = subjectDict

    def setExperimentalInfo(self, experDict: Dict):
        """
        Set experimental information.

        Args:
            experDict (Dict): Dictionary containing experimental information.
        """
        self.experimentalInfo = experDict

    def setModelInfo(self, modelDict: Dict):
        """
        Set model information.

        Args:
            modelDict (Dict): Dictionary containing biomechanical model information.
        """
        self.modelInfo = modelDict

    def setStpInfo(self, iDict: Dict):
        """
        Set additional information about spatiotemporal parameters.

        Args:
            iDict (Dict): Dictionary containing additional spatiotemporal information.
        """
        self.stpInfo = iDict

    def setKinematicInfo(self, iDict: Dict):
        """
        Set additional information about kinematic data.

        Args:
            iDict (Dict): Dictionary containing additional kinematic information.
        """
        self.kinematicInfo = iDict

    def setKineticInfo(self, iDict: Dict):
        """
        Set additional information about kinetic data.

        Args:
            iDict (Dict): Dictionary containing additional kinetic information.
        """
        self.kineticInfo = iDict

    def setEmgInfo(self, iDict: Dict):
        """
        Set additional information about EMG data.

        Args:
            iDict (Dict): Dictionary containing additional EMG information.
        """
        self.emgInfo = iDict

    def setScoreInfo(self, iDict: Dict):
        """
        Set additional information about scores.

        Args:
            iDict (Dict): Dictionary containing additional scoring information.
        """
        self.scoreInfo = iDict

    def setMuscleGeometryInfo(self, iDict: Dict):
        """
        Set additional information about muscle geometry.

        Args:
            iDict (Dict): Dictionary containing additional muscle geometry information.
        """
        self.muscleGeometryInfo = iDict

    def setMuscleDynamicInfo(self, iDict: Dict):
        """
        Set additional information about muscle dynamics.

        Args:
            iDict (Dict): Dictionary containing additional muscle dynamics information.
        """
        self.muscleDynamicInfo = iDict


    def getKinematicCycleNumbers(self) -> Tuple[int, int]:
        """
        Get the number of kinematic cycles for the left and right contexts.

        Returns:
            Tuple[int, int]: Tuple containing the number of left and right kinematic cycles.
        """
        for label, context in self.kinematicStats.data.keys():
            if context == "Left":
                n_leftCycles = self.kinematicStats.data[label, context]["values"].__len__(
                )
                break

        for label, context in self.kinematicStats.data.keys():
            if context == "Right":
                n_rightCycles = self.kinematicStats.data[label, context]["values"].__len__(
                )
                break
        return n_leftCycles, n_rightCycles

    def setCoactivation(self, labelEmg1: str, labelEmg2: str, context: str, res: Dict):
        """
        Set coactivation data for a pair of muscles in a specific context.

        Args:
            labelEmg1 (str): First EMG label.
            labelEmg2 (str): Second EMG label.
            context (str): Context of coactivation (e.g., 'Left', 'Right').
            res (Dict): Dictionary containing coactivation results.
        """
        self.coactivations[labelEmg1, labelEmg2, context] = res


# --- BUILDERS-----

class AbstractBuilder(object):
    def __init__(self, cycles=None):
        self.m_cycles = cycles

    def computeSpatioTemporel(self):
        pass

    def computeKinematics(self):
        pass

    def computeKinetics(self, momentContribution=False):
        pass

    def computeEmgEnvelopes(self):
        pass


class AnalysisBuilder(AbstractBuilder):
    """Analysis builder

    Unlike `GaitAnalysisBuilder`, this builder does not compute spatiotemporal parameters


    Args:
        cycles (Cycles): Cycles instance built from `CycleFilter`.
        kinematicLabelsDict (Optional[Dict[str, List[str]]]): Dictionary grouping kinematic output labels.
        kineticLabelsDict (Optional[Dict[str, List[str]]]): Dictionary grouping kinetic output labels.
        pointlabelSuffix (Optional[str]): Suffix for kinematic and kinetic labels.
        emgLabelList (Optional[List[str]]): List of EMG labels.
        geometryMuscleLabelsDict (Optional[Dict[str, List[str]]]): Dictionary grouping muscle geometry labels.
        dynamicMuscleLabelsDict (Optional[Dict[str, List[str]]]): Dictionary grouping muscle dynamic labels.
        modelInfos (Optional[Dict]): Information about the model.
        subjectInfos (Optional[Dict]): Information about the subject.
        experimentalInfos (Optional[Dict]): Information about the experimental conditions.
        emgs (Optional[Any]): Additional EMG data.


    **Notes**

        `modelInfos`,`experimentalInfos`, `subjectInfos` are simple dictionaries used to store basic information.
        When the analysis is exported as speadsheet, these informations appear as columns.

    """

    def __init__(self, cycles:Cycles,
                 kinematicLabelsDict:Optional[Dict[str, List[str]]]=None,
                 kineticLabelsDict:Optional[Dict[str, List[str]]]=None,
                 pointlabelSuffix:Optional[str]=None,
                 emgLabelList:Optional[List[str]]=None,
                 geometryMuscleLabelsDict:Optional[Dict[str, List[str]]]=None,
                 dynamicMuscleLabelsDict:Optional[Dict[str, List[str]]]=None,
                 modelInfos:Optional[Dict]=None, subjectInfos:Optional[Dict]=None, experimentalInfos:Optional[Dict]=None, emgs:Optional[Any]=None):


        super(AnalysisBuilder, self).__init__(cycles=cycles)

        self.m_kinematicLabelsDict = kinematicLabelsDict
        self.m_kineticLabelsDict = kineticLabelsDict
        self.m_pointlabelSuffix = pointlabelSuffix
        self.m_emgLabelList = emgLabelList
        self.m_emgs = emgs
        self.m_geometryMuscleLabelsDict = geometryMuscleLabelsDict
        self.m_dynamicMuscleLabelsDict = dynamicMuscleLabelsDict



    def computeSpatioTemporel(self):
        pass

    def computeKinematics(self):
        """
        Compute descriptive statistics of kinematic parameters.

        Returns:
            Tuple[Dict, Dict]: Tuple containing the output dictionary of kinematic data and the associated spatiotemporal parameters.
        """

        out = {}
        outPst = {}

        LOGGER.logger.info("--kinematic computation--")
        if self.m_cycles.kinematicCycles is not None:
            if "Left" in self.m_kinematicLabelsDict.keys():
                for label in self.m_kinematicLabelsDict["Left"]:
                    labelPlus = label + "_" + \
                        self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                    out[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                        self.m_cycles.kinematicCycles, labelPlus, "Left")

                LOGGER.logger.info("left kinematic computation---> done")
            else:
                LOGGER.logger.warning("No left Kinematic computation")

            if "Right" in self.m_kinematicLabelsDict.keys():
                for label in self.m_kinematicLabelsDict["Right"]:
                    labelPlus = label + "_" + \
                        self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                    out[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                        self.m_cycles.kinematicCycles, labelPlus, "Right")

                LOGGER.logger.info("right kinematic computation---> done")
            else:
                LOGGER.logger.warning("No right Kinematic computation")

        else:
            LOGGER.logger.warning("No Kinematic computation")

        return out, outPst

    def computeKinetics(self):
        """
        Compute descriptive statistics of kinetic parameters.

        Returns:
            Tuple[Dict, Dict, Dict]: Tuple containing the output dictionary of kinetic data, the associated spatiotemporal parameters, and optional additional data.
        """

        out = {}
        outPst = {}
        outOptional = {}

        LOGGER.logger.info("--kinetic computation--")
        if self.m_cycles.kineticCycles is not None:

            found_context = []
            for cycle in self.m_cycles.kineticCycles:
                found_context.append(cycle.context)

            if "Left" in self.m_kineticLabelsDict.keys():
                if "Left" in found_context:
                    for label in self.m_kineticLabelsDict["Left"]:
                        labelPlus = label + "_" + \
                           self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                        out[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                           self.m_cycles.kineticCycles, labelPlus, "Left")

                    if self.m_kinematicLabelsDict is not None:
                       for label in self.m_kinematicLabelsDict["Left"]:
                           labelPlus = label + "_" + \
                               self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                           outOptional[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                               self.m_cycles.kineticCycles, labelPlus, "Left")
                    LOGGER.logger.info("left kinetic computation---> done")
                else:
                    LOGGER.logger.warning("No left Kinetic computation")

            if "Right" in self.m_kineticLabelsDict.keys():
                if "Right" in found_context:
                    for label in self.m_kineticLabelsDict["Right"]:
                        labelPlus = label + "_" + \
                           self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                        out[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                           self.m_cycles.kineticCycles, labelPlus, "Right")

                    if self.m_kinematicLabelsDict is not None:
                        for label in self.m_kinematicLabelsDict["Right"]:
                            labelPlus = label + "_" + \
                               self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                            outOptional[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                               self.m_cycles.kineticCycles, labelPlus, "Right")

                    LOGGER.logger.info("right kinetic computation---> done")
                else:
                   LOGGER.logger.warning("No right Kinetic computation")

        else:
            LOGGER.logger.warning("No Kinetic computation")

        return out, outPst, outOptional

    def computeEmgEnvelopes(self):
        """
        Compute descriptive statistics of EMG envelopes.

        Returns:
            Tuple[Dict, Dict]: Tuple containing the output dictionary of EMG data and the associated spatiotemporal parameters.
        """
        out = {}
        outPst = {}

        LOGGER.logger.info("--emg computation--")
        if self.m_cycles.emgCycles is not None:
            for rawLabel in self.m_emgLabelList:
                out[rawLabel, "Left"] = CGM2cycle.analog_descriptiveStats(
                    self.m_cycles.emgCycles, rawLabel, "Left")
                out[rawLabel, "Right"] = CGM2cycle.analog_descriptiveStats(
                    self.m_cycles.emgCycles, rawLabel, "Right")

        else:
            LOGGER.logger.warning("No emg computation")

        return out, outPst

    def computeMuscleGeometry(self):
        """
        Compute descriptive statistics of muscle geometry parameters.

        Returns:
            Tuple[Dict, Dict]: Tuple containing the output dictionary of muscle geometry data and the associated spatiotemporal parameters.
        """

        out = {}
        outPst = {}

        LOGGER.logger.info("--kinematic computation--")
        if self.m_cycles.muscleGeometryCycles is not None:
            if "Left" in self.m_muscleLabelsDict.keys():
                for label in self.m_muscleLabelsDict["Left"]:
                    labelPlus = label + "_" + \
                        self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                    out[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                        self.m_cycles.muscleGeometryCycles, labelPlus, "Left")

                LOGGER.logger.info("left kinematic computation---> done")
            else:
                LOGGER.logger.warning("No left Kinematic computation")

            if "Right" in self.m_muscleLabelsDict.keys():
                for label in self.m_muscleLabelsDict["Right"]:
                    labelPlus = label + "_" + \
                        self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                    out[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                        self.m_cycles.muscleGeometryCycles, labelPlus, "Right")

                LOGGER.logger.info("right kinematic computation---> done")
            else:
                LOGGER.logger.warning("No right Kinematic computation")

        else:
            LOGGER.logger.warning("No Kinematic computation")

        return out, outPst

    def computeMuscleDynamic(self):
        """
        Compute descriptive statistics of muscle dynamic parameters.

        Returns:
            Tuple[Dict, Dict, Dict]: Tuple containing the output dictionary of muscle dynamic data, the associated spatiotemporal parameters, and optional additional data.
        """

        out = {}
        outPst = {}
        outOptional = {}

        LOGGER.logger.info("--kinetic computation--")
        if self.m_cycles.muscleDynamicCycles is not None:

           found_context = []
           for cycle in self.m_cycles.muscleDynamicCycles:
               found_context.append(cycle.context)

           if "Left" in self.m_muscleLabelsDict.keys():
               if "Left" in found_context:
                   for label in self.m_muscleLabelsDict["Left"]:
                       labelPlus = label + "_" + \
                           self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                       out[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                           self.m_cycles.muscleDynamicCycles, labelPlus, "Left")

                   if self.m_kinematicLabelsDict is not None:
                       for label in self.m_kinematicLabelsDict["Left"]:
                           labelPlus = label + "_" + \
                               self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                           outOptional[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                               self.m_cycles.muscleDynamicCycles, labelPlus, "Left")
                   LOGGER.logger.info("left kinetic computation---> done")
               else:
                   LOGGER.logger.warning("No left Kinetic computation")

           if "Right" in self.m_muscleLabelsDict.keys():
               if "Right" in found_context:
                   for label in self.m_muscleLabelsDict["Right"]:
                       labelPlus = label + "_" + \
                           self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                       out[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                           self.m_cycles.muscleDynamicCycles, labelPlus, "Right")

                   if self.m_kinematicLabelsDict is not None:
                       for label in self.m_kinematicLabelsDict["Right"]:
                           labelPlus = label + "_" + \
                               self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                           outOptional[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                               self.m_cycles.muscleDynamicCycles, labelPlus, "Right")

                   LOGGER.logger.info("right kinetic computation---> done")
               else:
                   LOGGER.logger.warning("No right Kinetic computation")

        else:
            LOGGER.logger.warning("No Kinetic computation")

        return out, outPst, outOptional

class GaitAnalysisBuilder(AbstractBuilder):
    """
    Builder for constructing a gait analysis instance.

    This builder computes spatiotemporal, kinematic, kinetic, and EMG parameters.

    Args:
        cycles (Cycles): Cycles instance built from `CycleFilter`.
        kinematicLabelsDict (Optional[Dict[str, List[str]]]): Dictionary grouping kinematic output labels.
        kineticLabelsDict (Optional[Dict[str, List[str]]]): Dictionary grouping kinetic output labels.
        pointlabelSuffix (Optional[str]): Suffix for kinematic and kinetic labels.
        emgLabelList (Optional[List[str]]): List of EMG labels.
        geometryMuscleLabelsDict (Optional[Dict[str, List[str]]]): Dictionary grouping muscle geometry labels.
        dynamicMuscleLabelsDict (Optional[Dict[str, List[str]]]): Dictionary grouping muscle dynamic labels.
        modelInfos (Optional[Dict]): Information about the model.
        subjectInfos (Optional[Dict]): Information about the subject.
        experimentalInfos (Optional[Dict]): Information about the experimental conditions.
        emgs (Optional[Any]): Additional EMG data.
    """
    

    def __init__(self, cycles:Cycles,
                 kinematicLabelsDict:Optional[Dict[str, List[str]]]=None,
                 kineticLabelsDict:Optional[Dict[str, List[str]]]=None,
                 pointlabelSuffix:Optional[str]=None,
                 emgLabelList:Optional[List[str]]=None,
                 geometryMuscleLabelsDict:Optional[Dict[str, List[str]]]=None,
                 dynamicMuscleLabelsDict:Optional[Dict[str, List[str]]]=None,
                 modelInfos:Optional[Dict]=None, subjectInfos:Optional[Dict]=None, experimentalInfos:Optional[Dict]=None, emgs:Optional[Any]=None):


        super(GaitAnalysisBuilder, self).__init__(cycles=cycles)

        self.m_kinematicLabelsDict = kinematicLabelsDict
        self.m_kineticLabelsDict = kineticLabelsDict
        self.m_pointlabelSuffix = pointlabelSuffix
        self.m_emgLabelList = emgLabelList
        self.m_geometryMuscleLabelsDict = geometryMuscleLabelsDict
        self.m_dynamicMuscleLabelsDict = dynamicMuscleLabelsDict

    def computeSpatioTemporel(self):
        """
        Compute spatiotemporal parameters. This method is currently not implemented.
        """
        out = {}

        LOGGER.logger.info("--stp computation--")
        if self.m_cycles.spatioTemporalCycles is not None:

            enableLeftComputation = len(
                [cycle for cycle in self.m_cycles.spatioTemporalCycles if cycle.enableFlag and cycle.context == "Left"])
            enableRightComputation = len(
                [cycle for cycle in self.m_cycles.spatioTemporalCycles if cycle.enableFlag and cycle.context == "Right"])

            for label in CGM2cycle.GaitCycle.STP_LABELS:
                if enableLeftComputation:
                    try:
                        out[label, "Left"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                            self.m_cycles.spatioTemporalCycles, label, "Left")
                    except KeyError:
                        LOGGER.logger.warning(
                            "the spatio temporal parameter [%s] is not computed for the left context" % (label))

                if enableRightComputation:
                    try:
                        out[label, "Right"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                            self.m_cycles.spatioTemporalCycles, label, "Right")
                    except KeyError:
                        LOGGER.logger.warning(
                            "the spatio temporal parameter [%s] is not computed for the right context" % (label))

            if enableLeftComputation:
                LOGGER.logger.info("left stp computation---> done")
            if enableRightComputation:
                LOGGER.logger.info("right stp computation---> done")
        else:
            LOGGER.logger.warning("No spatioTemporal computation")

        return out

    def computeKinematics(self):
        """
        Compute descriptive statistics of kinematic parameters.

        Returns:
            Tuple[Dict, Dict]: Tuple containing the output dictionary of kinematic data and the associated spatiotemporal parameters.
        """

        out = {}
        outPst = {}

        LOGGER.logger.info("--kinematic computation--")
        if self.m_cycles.kinematicCycles is not None:
            if "Left" in self.m_kinematicLabelsDict.keys():
                for label in self.m_kinematicLabelsDict["Left"]:
                    labelPlus = label + "_" + \
                        self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                    out[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                        self.m_cycles.kinematicCycles, labelPlus, "Left")

                for label in CGM2cycle.GaitCycle.STP_LABELS:
                    outPst[label, "Left"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                        self.m_cycles.kinematicCycles, label, "Left")

                LOGGER.logger.info("left kinematic computation---> done")
            else:
                LOGGER.logger.warning("No left Kinematic computation")

            if "Right" in self.m_kinematicLabelsDict.keys():
                for label in self.m_kinematicLabelsDict["Right"]:
                    labelPlus = label + "_" + \
                        self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                    out[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                        self.m_cycles.kinematicCycles, labelPlus, "Right")

                for label in CGM2cycle.GaitCycle.STP_LABELS:
                    outPst[label, "Right"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                        self.m_cycles.kinematicCycles, label, "Right")

                LOGGER.logger.info("right kinematic computation---> done")
            else:
                LOGGER.logger.warning("No right Kinematic computation")

        else:
            LOGGER.logger.warning("No Kinematic computation")

        return out, outPst

    def computeKinetics(self):
        """
        Compute descriptive statistics of kinetic parameters.

        Returns:
            Tuple[Dict, Dict, Dict]: Tuple containing the output dictionary of kinetic data, the associated spatiotemporal parameters, and optional additional data.
        """

        out = {}
        outPst = {}
        outOptional = {}

        LOGGER.logger.info("--kinetic computation--")
        if self.m_cycles.kineticCycles is not None:

           found_context = []
           for cycle in self.m_cycles.kineticCycles:
               found_context.append(cycle.context)

           if "Left" in self.m_kineticLabelsDict.keys():
               if "Left" in found_context:
                   for label in self.m_kineticLabelsDict["Left"]:
                       labelPlus = label + "_" + \
                           self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                       out[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                           self.m_cycles.kineticCycles, labelPlus, "Left")
                   for label in CGM2cycle.GaitCycle.STP_LABELS:
                       outPst[label, "Left"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                           self.m_cycles.kineticCycles, label, "Left")

                   if self.m_kinematicLabelsDict is not None:
                       for label in self.m_kinematicLabelsDict["Left"]:
                           labelPlus = label + "_" + \
                               self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                           outOptional[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                               self.m_cycles.kineticCycles, labelPlus, "Left")
                   LOGGER.logger.info("left kinetic computation---> done")
               else:
                   LOGGER.logger.warning("No left Kinetic computation")

           if "Right" in self.m_kineticLabelsDict.keys():
               if "Right" in found_context:
                   for label in self.m_kineticLabelsDict["Right"]:
                       labelPlus = label + "_" + \
                           self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                       out[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                           self.m_cycles.kineticCycles, labelPlus, "Right")

                   for label in CGM2cycle.GaitCycle.STP_LABELS:
                       outPst[label, "Right"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                           self.m_cycles.kineticCycles, label, "Right")

                   if self.m_kinematicLabelsDict is not None:
                       for label in self.m_kinematicLabelsDict["Right"]:
                           labelPlus = label + "_" + \
                               self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                           outOptional[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                               self.m_cycles.kineticCycles, labelPlus, "Right")

                   LOGGER.logger.info("right kinetic computation---> done")
               else:
                   LOGGER.logger.warning("No right Kinetic computation")

        else:
            LOGGER.logger.warning("No Kinetic computation")

        return out, outPst, outOptional

    def computeEmgEnvelopes(self):
        """
        Compute descriptive statistics of EMG envelopes.

        Returns:
            Tuple[Dict, Dict]: Tuple containing the output dictionary of EMG data and the associated spatiotemporal parameters.
        """

        out = {}
        outPst = {}
        LOGGER.logger.info("--emg computation--")
        if self.m_cycles.emgCycles is not None:

            for rawLabel in self.m_emgLabelList:
                out[rawLabel, "Left"] = CGM2cycle.analog_descriptiveStats(
                    self.m_cycles.emgCycles, rawLabel, "Left")
                out[rawLabel, "Right"] = CGM2cycle.analog_descriptiveStats(
                    self.m_cycles.emgCycles, rawLabel, "Right")

            for label in CGM2cycle.GaitCycle.STP_LABELS:
                try:
                    outPst[label, "Left"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                        self.m_cycles.emgCycles, label, "Left")
                    outPst[label, "Right"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                        self.m_cycles.emgCycles, label, "Right")
                except KeyError:
                    LOGGER.logger.warning(
                        "the spatio temporal parameter [%s] is not computed" % (label))

        else:
            LOGGER.logger.warning("No emg computation")

        return out, outPst

    def computeMuscleGeometry(self):
        """
        Compute descriptive statistics of muscle geometry parameters.

        Returns:
            Tuple[Dict, Dict]: Tuple containing the output dictionary of muscle geometry data and the associated spatiotemporal parameters.
        """

        out = {}
        outPst = {}

        LOGGER.logger.info("--Muscle Geometry computation--")
        if self.m_cycles.muscleGeometryCycles is not None:
            if "Left" in self.m_geometryMuscleLabelsDict.keys():
                for label in self.m_geometryMuscleLabelsDict["Left"]:
                    labelPlus = label + "_" + \
                        self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                    out[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                        self.m_cycles.muscleGeometryCycles, labelPlus, "Left")

                for label in CGM2cycle.GaitCycle.STP_LABELS:
                    outPst[label, "Left"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                        self.m_cycles.muscleGeometryCycles, label, "Left")

                LOGGER.logger.info("left kinematic computation---> done")
            else:
                LOGGER.logger.warning("No left Kinematic computation")

            if "Right" in self.m_geometryMuscleLabelsDict.keys():
                for label in self.m_geometryMuscleLabelsDict["Right"]:
                    labelPlus = label + "_" + \
                        self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                    out[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                        self.m_cycles.muscleGeometryCycles, labelPlus, "Right")

                for label in CGM2cycle.GaitCycle.STP_LABELS:
                    outPst[label, "Right"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                        self.m_cycles.muscleGeometryCycles, label, "Right")

                LOGGER.logger.info("right kinematic computation---> done")
            else:
                LOGGER.logger.warning("No right Kinematic computation")

        else:
            LOGGER.logger.warning("No Kinematic computation")

        return out, outPst

    def computeMuscleDynamic(self):
        """
        Compute descriptive statistics of muscle dynamic parameters.

        Returns:
            Tuple[Dict, Dict, Dict]: Tuple containing the output dictionary of muscle dynamic data, the associated spatiotemporal parameters, and optional additional data.
        """

        out = {}
        outPst = {}
        outOptional = {}

        LOGGER.logger.info("--kinetic computation--")
        if self.m_cycles.muscleDynamicCycles is not None:

           found_context = []
           for cycle in self.m_cycles.muscleDynamicCycles:
               found_context.append(cycle.context)

           if "Left" in self.m_muscleLabelsDict.keys():
               if "Left" in found_context:
                   for label in self.m_muscleLabelsDict["Left"]:
                       labelPlus = label + "_" + \
                           self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                       out[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                           self.m_cycles.muscleDynamicCycles, labelPlus, "Left")
                   for label in CGM2cycle.GaitCycle.STP_LABELS:
                       outPst[label, "Left"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                           self.m_cycles.muscleDynamicCycles, label, "Left")

                   if self.m_kinematicLabelsDict is not None:
                       for label in self.m_kinematicLabelsDict["Left"]:
                           labelPlus = label + "_" + \
                               self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                           outOptional[labelPlus, "Left"] = CGM2cycle.point_descriptiveStats(
                               self.m_cycles.muscleDynamicCycles, labelPlus, "Left")
                   LOGGER.logger.info("left kinetic computation---> done")
               else:
                   LOGGER.logger.warning("No left Kinetic computation")

           if "Right" in self.m_muscleLabelsDict.keys():
               if "Right" in found_context:
                   for label in self.m_muscleLabelsDict["Right"]:
                       labelPlus = label + "_" + \
                           self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                       out[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                           self.m_cycles.muscleDynamicCycles, labelPlus, "Right")

                   for label in CGM2cycle.GaitCycle.STP_LABELS:
                       outPst[label, "Right"] = CGM2cycle.spatioTemporelParameter_descriptiveStats(
                           self.m_cycles.muscleDynamicCycles, label, "Right")

                   if self.m_kinematicLabelsDict is not None:
                       for label in self.m_kinematicLabelsDict["Right"]:
                           labelPlus = label + "_" + \
                               self.m_pointlabelSuffix if self.m_pointlabelSuffix is not None else label
                           outOptional[labelPlus, "Right"] = CGM2cycle.point_descriptiveStats(
                               self.m_cycles.muscleDynamicCycles, labelPlus, "Right")

                   LOGGER.logger.info("right kinetic computation---> done")
               else:
                   LOGGER.logger.warning("No right Kinetic computation")

        else:
            LOGGER.logger.warning("No Kinetic computation")

        return out, outPst, outOptional

# ---- FILTERS -----


class AnalysisFilter(object):
    """ Filter building an `Analysis` instance.
    """

    def __init__(self):
        self.__concreteAnalysisBuilder = None
        self.analysis = Analysis()

        self.subjectInfo = None
        self.experimentalInfo = None
        self.modelInfo = None

    def setBuilder(self, concreteBuilder):
        """Set a concrete builder

        Args:
            concreteBuilder (Builder) - a concrete Builder

        """

        self.__concreteAnalysisBuilder = concreteBuilder

    def setInfo(self, subject=None, experimental=None, model=None):
        """Set informations

        Args:
            subject (dict,Optional[None]): subject info
            experimental (dict,Optional[None]): xperimental info
            model (dict,Optional[None]): model info

        """

        if subject is not None:
            self.subjectInfo = subject

        if experimental is not None:
            self.experimentalInfo = experimental

        if model is not None:
            self.modelInfo = model


    def build(self):
        """ Run the filter and build the analysis
        """
        pstOut = self.__concreteAnalysisBuilder.computeSpatioTemporel()
        self.analysis.setStp(pstOut)

        kinematicOut, matchPst_kinematic = self.__concreteAnalysisBuilder.computeKinematics()
        self.analysis.setKinematic(kinematicOut, pst=matchPst_kinematic)

        kineticOut, matchPst_kinetic, matchKinematic = self.__concreteAnalysisBuilder.computeKinetics()
        self.analysis.setKinetic(
            kineticOut, pst=matchPst_kinetic, optionalData=matchKinematic)

        if self.__concreteAnalysisBuilder.m_emgLabelList:
            emgOut, matchPst_emg = self.__concreteAnalysisBuilder.computeEmgEnvelopes()
            self.analysis.setEmg(emgOut, pst=matchPst_emg)

        if self.__concreteAnalysisBuilder.m_geometryMuscleLabelsDict:
            muscleGeoOut, matchPst_muscleGeo = self.__concreteAnalysisBuilder.computeMuscleGeometry()
            self.analysis.setMuscleGeometry(muscleGeoOut, pst=matchPst_muscleGeo)

        if self.__concreteAnalysisBuilder.m_dynamicMuscleLabelsDict:
            muscleDynOut, matchPst_muscleDyn = self.__concreteAnalysisBuilder.computeMuscleDynamic()
            self.analysis.setMuscleDynamic(muscleDynOut, pst=matchPst_muscleDyn)


        if self.subjectInfo is not None:
            self.analysis.setSubjectInfo(self.subjectInfo)

        if self.experimentalInfo is not None:
            self.analysis.setExperimentalInfo(self.experimentalInfo)

        if self.modelInfo is not None:
            self.analysis.setModelInfo(self.modelInfo)
