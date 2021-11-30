# -*- coding: utf-8 -*-
import pyCGM2.Processing.cycle as CGM2cycle
import pyCGM2
LOGGER = pyCGM2.LOGGER


class AnalysisStructure:
    data = dict()
    pst = dict()
    optionalData = dict()


# ---- PATTERN BUILDER ------

# --- OBJECT TO BUILD-----
class Analysis():
    """
       Object built from AnalysisFilter.build().

       Analysis work as **class-container**. Its attribute members return descriptive statistics

       Attributes :

          - `stpStats` (dict)  - descritive statictics of stapiotemporal parameters
          - `kinematicStats` (AnalysisStructure)  - descritive statictics of kinematics data.
          - `kineticStats` (AnalysisStructure)  - descritive statictics of kinetics data.
          - `emgStats` (AnalysisStructure)  - descritive statictics of emg data.

       .. note:

           Notice kinematicStats, kineticStats and emgStats are `AnalysisStructure object`. This object implies two dictionnary as sublevels.

             - `data` collect descriptive statistics of either kinematics, kinetics or emg.
             - `pst` returns the spatiotemporal parameters of all cycles involved in either kinematics, kinetics or emg processing.


    """

    def __init__(self):

        self.stpStats = dict()
        self.kinematicStats = AnalysisStructure()
        self.kineticStats = AnalysisStructure()
        self.emgStats = AnalysisStructure()
        self.gps = None
        self.gvs = None
        self.coactivations = dict()
        self.subjectInfo = None
        self.experimentalInfo = None
        self.modelInfo = None

        self.emgInfo = None
        self.kinematicInfo = None
        self.kineticInfo = None
        self.stpInfo = None
        self.scoreInfo = None

    def setStp(self, inDict):
        self.stpStats = inDict

    def setKinematic(self, data, pst=dict()):
        self.kinematicStats.data = data
        self.kinematicStats.pst = pst

    def setKinetic(self, data, pst=dict(), optionalData=dict()):
        self.kineticStats.data = data
        self.kineticStats.pst = pst
        self.kineticStats.optionalData = optionalData

    def setEmg(self, data, pst=dict()):
        self.emgStats.data = data
        self.emgStats.pst = pst

    def setGps(self, GpsStatsOverall, GpsStatsContext):
        self.gps = dict()
        self.gps["Overall"] = GpsStatsOverall
        self.gps["Context"] = GpsStatsContext

    def setGvs(self, gvsStats):
        self.gvs = gvsStats

    def setSubjectInfo(self, subjectDict):
        self.subjectInfo = subjectDict

    def setExperimentalInfo(self, experDict):
        self.experimentalInfo = experDict

    def setModelInfo(self, modeltDict):
        self.modelInfo = modeltDict

    def setStpInfo(self, iDict):
        self.stpInfo = iDict

    def setKinematicInfo(self, iDict):
        self.kinematicInfo = iDict

    def setKineticInfo(self, iDict):
        self.kineticInfo = iDict

    def setEmgInfo(self, iDict):
        self.emgInfo = iDict

    def setScoreInfo(self, iDict):
        self.scoreInfo = iDict

    def getKinematicCycleNumbers(self):
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

    def setCoactivation(self, labelEmg1, labelEmg2, context, res):
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
    """
        **Description** :
    """

    def __init__(self, cycles,
                 kinematicLabelsDict=None,
                 kineticLabelsDict=None,
                 pointlabelSuffix=None,
                 emgLabelList=None,
                 modelInfos=None, subjectInfos=None, experimentalInfos=None, emgs=None):
        """
            :Parameters:
                 - `cycles` (pyCGM2.Processing.cycle.Cycles) - Cycles instance built from CycleFilter
                 - `kinematicLabelsDict` (dict) - dictionnary with two items (Left and Right) grouping kinematic output label
                 - `kineticLabelsDict` (dict) - dictionnary with two items (Left and Right) grouping kinetic output label
                 - `pointlabelSuffix` (dict) - suffix ending kinematicLabels and kineticLabels dictionnaries
                 - `emgLabelList` (list of str) - labels of used emg
                 - `subjectInfos` (dict) - information about the subject
                 - `modelInfos` (dict) - information about the model
                 - `experimentalInfos` (dict) - information about the experimental conditions
                 -  .. attention:: `emgs` (pyCGM2emg) - object in progress


            .. note::

                modelInfos,experimentalInfos, subjectInfos are convenient dictionnaries in which you can store different sort of information






        """

        super(AnalysisBuilder, self).__init__(cycles=cycles)

        self.m_kinematicLabelsDict = kinematicLabelsDict
        self.m_kineticLabelsDict = kineticLabelsDict
        self.m_pointlabelSuffix = pointlabelSuffix
        self.m_emgLabelList = emgLabelList
        self.m_emgs = emgs

    def computeSpatioTemporel(self):
        pass

    def computeKinematics(self):
        """ compute descriptive of kinematics parameters

            :return:
                - `out` (dict) - dictionnary with descriptive statictics of kinematics parameters
                - `outPst` ( dict) - dictionnary with descriptive statictics of spatio-temporal parameters matching  kinematics parameters

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
        """ compute descriptive of kinetics parameters

            :return:
                - `out` (dict) - dictionnary with descriptive statictics of kinetics parameters
                - `outPst` ( dict) - dictionnary with descriptive statictics of spatio-temporal parameters matching  kinetics parameters

        """

        out = {}
        outPst = {}
        outOptional = {}

        LOGGER.logger.info("--kinetic computation--")
        if self.m_cycles.kineticCycles is not None:

           found_context = list()
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
            Compute descriptive of emg values

            :return:
                - `out` (dict) - dictionnary with descriptive statictics of emg envelopes
                - `outPst` ( dict) - dictionnary with descriptive statictics of spatio-temporal parameters matching emg envelopes
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


class GaitAnalysisBuilder(AbstractBuilder):
    """
        **Description** : builder of a common clinical gait analysis
    """

    def __init__(self, cycles,
                 kinematicLabelsDict=None,
                 kineticLabelsDict=None,
                 pointlabelSuffix=None,
                 emgLabelList=None,
                 modelInfos=None, subjectInfos=None, experimentalInfos=None, emgs=None):
        """
            :Parameters:
                 - `cycles` (pyCGM2.Processing.cycle.Cycles) - Cycles instance built from CycleFilter
                 - `kinematicLabelsDict` (dict) - dictionnary with two items (Left and Right) grouping kinematic output label
                 - `kineticLabelsDict` (dict) - dictionnary with two items (Left and Right) grouping kinetic output label
                 - `pointlabelSuffix` (dict) - suffix ending kinematicLabels and kineticLabels dictionnaries
                 - `emgLabelList` (list of str) - labels of used emg
                 - `subjectInfos` (dict) - information about the subject
                 - `modelInfos` (dict) - information about the model
                 - `experimentalInfos` (dict) - information about the experimental conditions
                 -  .. attention:: `emgs` (pyCGM2emg) - object in progress


            .. note::

                modelInfos,experimentalInfos, subjectInfos are convenient dictionnaries in which you can store different sort of information






        """

        super(GaitAnalysisBuilder, self).__init__(cycles=cycles)

        self.m_kinematicLabelsDict = kinematicLabelsDict
        self.m_kineticLabelsDict = kineticLabelsDict
        self.m_pointlabelSuffix = pointlabelSuffix
        self.m_emgLabelList = emgLabelList

    def computeSpatioTemporel(self):
        """
            **Description:** compute descriptive of spatio-temporal parameters

            :return:
                - `out` (dict) - dictionnary with descriptive statictics of spatio-temporal parameters

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
        """ compute descriptive of kinematics parameters

            :return:
                - `out` (dict) - dictionnary with descriptive statictics of kinematics parameters
                - `outPst` ( dict) - dictionnary with descriptive statictics of spatio-temporal parameters matching  kinematics parameters

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
        """ compute descriptive of kinetics parameters

            :return:
                - `out` (dict) - dictionnary with descriptive statictics of kinetics parameters
                - `outPst` ( dict) - dictionnary with descriptive statictics of spatio-temporal parameters matching  kinetics parameters

        """

        out = {}
        outPst = {}
        outOptional = {}

        LOGGER.logger.info("--kinetic computation--")
        if self.m_cycles.kineticCycles is not None:

           found_context = list()
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
            Compute descriptive of emg values

            :return:
                - `out` (dict) - dictionnary with descriptive statictics of emg envelopes
                - `outPst` ( dict) - dictionnary with descriptive statictics of spatio-temporal parameters matching emg envelopes
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

# ---- FILTERS -----


class AnalysisFilter(object):
    """
         Filter building an Analysis instance.
    """

    def __init__(self):
        self.__concreteAnalysisBuilder = None
        self.analysis = Analysis()

        self.subjectInfo = None
        self.experimentalInfo = None
        self.modelInfo = None

    def setBuilder(self, concreteBuilder):
        """
             set a concrete builder

            :Parameters:
                - `concreteBuilder` (Builder) - a concrete Builder

        """

        self.__concreteAnalysisBuilder = concreteBuilder

    def setInfo(self, subject=None, experimental=None, model=None):

        if subject is not None:
            self.subjectInfo = subject

        if experimental is not None:
            self.experimentalInfo = experimental

        if model is not None:
            self.modelInfo = model

        # self.stpInfo = dict()
        # self.kinematicInfo = dict()
        # self.kineticInfo = dict()
        # self.emgInfo = dict()

    def build(self):
        """
            build member analysis from a concrete builder

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

        if self.subjectInfo is not None:
            self.analysis.setSubjectInfo(self.subjectInfo)

        if self.experimentalInfo is not None:
            self.analysis.setExperimentalInfo(self.experimentalInfo)

        if self.modelInfo is not None:
            self.analysis.setModelInfo(self.modelInfo)
