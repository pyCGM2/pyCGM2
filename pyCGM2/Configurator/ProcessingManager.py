# -*- coding: utf-8 -*-
import pyCGM2
from pyCGM2.Configurator import Manager
from pyCGM2.Utils import files
from pyCGM2 import enums
import pyCGM2; LOGGER = pyCGM2.LOGGER
import copy
from pyCGM2.Report import normativeDatasets


class GaitProcessingConfigManager(Manager.ConfigManager):
    """

    """
    def __init__(self,settings, modelVersion = None):
        super(GaitProcessingConfigManager, self).__init__(settings)
        self.modelVersion = modelVersion

    @property
    def task(self):
        return self._userSettings["Task"]

    @property
    def conditions(self):
        return self._userSettings["Conditions"]


    @property
    def experimentalInfo(self):
        return self._userSettings["ExperimentalInfo"]

    @property
    def subjectInfo(self):
        return self._userSettings["SubjectInfo"]

    @property
    def modelledTrials(self):
        return self._userSettings["ModelledTrials"]


    @property
    def normativeData(self):

        normativeData = {"Author" : self._userSettings["NormativeData"]["Author"],
                        "Modality" : self._userSettings["NormativeData"]["Modality"]}

        if normativeData["Author"] == "Schwartz2008":
            chosenModality = normativeData["Modality"]
            nds = normativeDatasets.Schwartz2008(chosenModality)    # modalites : "Very Slow" ,"Slow", "Free", "Fast", "Very Fast"
        elif normativeData["Author"] == "Pinzone2014":
            chosenModality = normativeData["Modality"]
            nds = normativeDatasets.Pinzone2014(chosenModality)

        return nds

    @property
    def pointSuffix(self):
        value = self._userSettings["Point suffix"]
        return  None if value=="None" else value

    @property
    def consistencyFlag(self):
        return self._userSettings["Consistency"]

    @property
    def title(self):
        if self._userSettings["Title"] == "None":

            if self.modelVersion is not None:
                title = self.modelVersion +"-"+self._userSettings["Task"]+ "-Gait Processing"
            else:
                title = self._userSettings["Task"]+ "-Gait Processing"

            self._userSettings["Title"] = title
            return self._userSettings["Title"]
        else:
            return self._userSettings["Title"]

    @property
    def bodyPart(self):
        return self._userSettings["BodyPart"]
