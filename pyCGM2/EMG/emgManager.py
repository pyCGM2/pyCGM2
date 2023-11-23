
from pyCGM2.Utils import files
import os

import pyCGM2
LOGGER = pyCGM2.LOGGER
from typing import Optional, Union, Dict

class EmgManager(object):
    """
    Class to manage  emg settings ( ie the emg.settings file)

    Args:
        DATA_PATH (str): data folder path
        emgSettings (str): filename with emg settings 
    """

    def __init__(self, DATA_PATH:str, emgSettings: Optional[Union[str, Dict]]=None):

        if emgSettings is None:
            if os.path.isfile(DATA_PATH + "emg.settings"):
                emgSettings = files.openFile(DATA_PATH, "emg.settings")
                LOGGER.logger.info(
                    "[pyCGM2]: emg.settings detected in the data folder")
            else:
                emgSettings = files.openFile(
                    pyCGM2.PYCGM2_SETTINGS_FOLDER, "emg.settings")
        else:
            if isinstance(emgSettings,str):
                if DATA_PATH is not None:
                    LOGGER.logger.info( f"[pyCGM2]: emg settings loaded from => {emgSettings} ")
                    emgSettings = files.openFile(DATA_PATH, emgSettings)
                else: 
                    emgSettings = files.openFile(None, emgSettings)
            else:
                pass

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

        out = []
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(channel)

        return out

    def getMuscles(self):
        """ return the muscles """
        out = []
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(self.m_emgChannelSection[channel]["Muscle"])

        return out

    def getSides(self):
        """ return side of each emg"""
        out = []
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(self.m_emgChannelSection[channel]["Context"])
        return out

    def getNormalActivity(self):
        """ return the normal activity muscle reference
        """
        out = []
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(self.m_emgChannelSection[channel]["NormalActivity"])

        return out
