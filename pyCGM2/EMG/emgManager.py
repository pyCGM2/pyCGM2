
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
        emgSettings (Optional[Union[str, Dict]]): Filename or (path+filename) with EMG settings or a dictionary of EMG settings. 
        If None, default settings are loaded.
    """

    def __init__(self, DATA_PATH:str, emgSettings: Optional[Union[str, Dict]]=None):
        """ Initializes the EmgManager with a path to the data folder and EMG settings"""
        
        settings = files.openFile(
                    pyCGM2.PYCGM2_SETTINGS_FOLDER, "emg.settings")

        if DATA_PATH is not None and isinstance(emgSettings,str):
            LOGGER.logger.info( f"[pyCGM2] - emgsettings loaded from {DATA_PATH+emgSettings}")
            settings = files.openFile(DATA_PATH, emgSettings)

        elif DATA_PATH is not None and emgSettings is None:
            if os.path.isfile(DATA_PATH + "emg.settings"):
                LOGGER.logger.info( f"[pyCGM2] - local emgsettings detected in {DATA_PATH}")
                settings = files.openFile(DATA_PATH, "emg.settings")
                    
        elif DATA_PATH is None and isinstance(emgSettings,str):
            settings = files.openFile(None, emgSettings)
            LOGGER.logger.info( f"[pyCGM2] - emgsettings loaded from {emgSettings}")

        elif DATA_PATH is None and isinstance(emgSettings,Dict):
            LOGGER.logger.info( f"[pyCGM2] - emg settings loaded from a dictionnary")
            settings = emgSettings
        else:
            LOGGER.logger.info( f"[pyCGM2] - default emgsettings")

        
    
        self.m_emgSettings = settings
        self.m_emgChannelSection = settings["CHANNELS"]
        self.m_emgProcessingSection = settings["Processing"]

       

        self._emg={}
        for key in self.m_emgChannelSection:
            if self.m_emgChannelSection[key]["Muscle"] is not None: 
                self._emg[key] = self.m_emgChannelSection[key]["Muscle"]+"_"+self.m_emgChannelSection[key]["Context"]


        # self._emg=[]
        # for i in range(0,len(self.m_labels)):
        #     self.combinedEMG.append([self.m_labels[i],self.m_side[i], self.m_muscles[i]])
   
    
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
    
    def getChannel(self,muscle:str,eventContext:str):
        """
        Return the EMG channel label from a defined muscle and eventContext.

        Returns:
            str: the emg hannel.
        """
        for key, value in self._emg.items():
            if value == muscle+"_"+eventContext:
                return key
        return None  # Si la valeur n'est pas trouvée


