
from pyCGM2.Utils import files
import os

import pyCGM2
LOGGER = pyCGM2.LOGGER
from typing import Optional, Union, Dict

class EmgManager(object):
    """
    Class to manage EMG settings (i.e., the emg.settings file).

    This class is designed to handle the configuration and retrieval of EMG settings, which include channel information and processing parameters.

    Args:
        DATA_PATH (str): Data folder path where the EMG settings file is located.
        emgSettings (Optional[Union[str, Dict]]): Filename with EMG settings or a dictionary of EMG settings. If None, default settings are loaded.
    """

    def __init__(self, DATA_PATH:str, emgSettings: Optional[Union[str, Dict]]=None):
        """ Initializes the EmgManager with a path to the data folder and EMG settings"""
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
        """
        Return the channel section of the EMG settings.

        This section typically contains configuration related to EMG channels.

        Returns:
            Dict: The channel section of the EMG settings.
        """
        return self.m_emgChannelSection

    def getProcessingSection(self):
        """
        Return the processing section of the EMG settings.

        This section usually includes parameters related to EMG signal processing.

        Returns:
            Dict: The processing section of the EMG settings.
        """
        return self.m_emgProcessingSection

    def getChannels(self):
        """
        Return the EMG channel labels from the settings.

        Filters out channels that are not associated with a muscle.

        Returns:
            List[str]: List of EMG channel labels.
        """

        out = []
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(channel)

        return out

    def getMuscles(self):
        """
        Return the muscle names associated with EMG channels.

        Filters out channels that are not associated with a muscle.

        Returns:
            List[str]: List of muscle names corresponding to EMG channels.
        """
        out = []
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(self.m_emgChannelSection[channel]["Muscle"])

        return out

    def getSides(self):
        """
        Return the side (context) of each EMG channel.

        Filters out channels that are not associated with a muscle.

        Returns:
            List[str]: List of sides (e.g., 'Left', 'Right') for each EMG channel.
        """
        out = []
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(self.m_emgChannelSection[channel]["Context"])
        return out

    def getNormalActivity(self):
        """
        Return the normal activity muscle reference from the EMG settings.

        This information is used for normalization purposes during EMG processing.

        Returns:
            List[str]: List of normal activities for each muscle.
        """
        out = []
        for channel in self.m_emgChannelSection.keys():
            if self.m_emgChannelSection[channel]["Muscle"] is not None and self.m_emgChannelSection[channel]["Muscle"] != "None":
                out.append(self.m_emgChannelSection[channel]["NormalActivity"])

        return out
