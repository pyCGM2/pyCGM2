# -*- coding: utf-8 -*-
import os

import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files


class EmgManager(object):
    """

    """

    def __init__(self,DATA_PATH,emgSettings=None):

        if emgSettings is None:
            if os.path.isfile(DATA_PATH + "emg.settings"):
                emgSettings = files.openFile(DATA_PATH,"emg.settings")
                LOGGER.logger.warning("[pyCGM2]: emg.settings detected in the data folder")
            else:
                emgSettings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"emg.settings")


        self.m_emgSettings = emgSettings

        self.m_emgChannelSection = emgSettings["CHANNELS"]
        self.m_emgProcessingSection = emgSettings["Processing"]

    def getChannelSection(self):
        return self.m_emgChannelSection

    def getProcessingSection(self):
        return self.m_emgProcessingSection

    def getChannels(self):

        out = list()
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None" :
                out.append(channel)

        return out

    def getMuscles(self):
        out = list()
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None" :
                out.append(self.m_emgChannelSection[channel]["Muscle"])

        return out

    def getSides(self):
        out = list()
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None" :
                out.append(self.m_emgChannelSection[channel]["Context"])
        return out



    def getNormalActivity(self):
        out = list()
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None" :
                out.append(self.m_emgChannelSection[channel]["NormalActivity"])

        return out
