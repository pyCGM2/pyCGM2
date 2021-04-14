# -*- coding: utf-8 -*-
from pyCGM2.Configurator import EmgManager
from pyCGM2.Configurator import ModelManager

# -----emg------
def getEmgConfiguration(userSettings,internalSettings):
    """
    return emg settings

    :param userSettings [dict]:  content of the userSettings yaml file
    :param internalSettings [dict]: content of the internalSettings yaml file

    **Return**
    :param [dict]:  eventual emg settings

    """
    # --- Manager ----
    manager = EmgManager.EmgConfigManager(userSettings,localInternalSettings=internalSettings)
    manager.contruct()

    return manager.getEmgConfiguration()
