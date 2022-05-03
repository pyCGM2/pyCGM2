# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/EMG
#APIDOC["Draft"]=False
#--end--

from pyCGM2.Utils import files
import os

import pyCGM2
LOGGER = pyCGM2.LOGGER


class EmgManager(object):
    """
    Class to manage  emg settings ( ie the emg.settings file)

    Args:
        DATA_PATH (str): data folder path
        emgSettings (str,Optional[None]): content of the emg.settings file

    """

    def __init__(self, DATA_PATH, emgSettings=None):


        if emgSettings is None:
            if os.path.isfile(DATA_PATH + "emg.settings"):
                emgSettings = files.openFile(DATA_PATH, "emg.settings")
                LOGGER.logger.warning(
                    "[pyCGM2]: emg.settings detected in the data folder")
            else:
                emgSettings = files.openFile(
                    pyCGM2.PYCGM2_SETTINGS_FOLDER, "emg.settings")

        self.m_emgSettings = emgSettings

        self.m_emgChannelSection = emgSettings["CHANNELS"]
        self.m_emgProcessingSection = emgSettings["Processing"]

    def getChannelSection(self):
        """ return the channel section of the emg settings """
        return self.m_emgChannelSection

    def getProcessingSection(self):
        """ return the processing section of the emg settings """
        return self.m_emgProcessingSection

    def getChannels(self):
        """ return the channel labels  """

        out = list()
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(channel)

        return out

    def getMuscles(self):
        """ return the muscles """
        out = list()
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(self.m_emgChannelSection[channel]["Muscle"])

        return out

    def getSides(self):
        """ return side of each emg"""
        out = list()
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(self.m_emgChannelSection[channel]["Context"])
        return out

    def getNormalActivity(self):
        """ return the normal activity muscle reference
        """
        out = list()
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(self.m_emgChannelSection[channel]["NormalActivity"])

        return out
